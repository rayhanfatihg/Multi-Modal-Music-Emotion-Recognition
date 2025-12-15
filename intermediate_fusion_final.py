import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'target_dim': 128,
    'dropout': 0.5,
    'lr': 1e-3,              
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 60,            # Increased epochs for Mixup
    'n_splits': 5,
    'noise_level': 0.05,
    'modality_dropout': 0.2,
    'T_0': 10,               
    'T_mult': 2,
    'mixup_alpha': 0.4,      # Mixup Beta distribution param
    'label_smoothing': 0.1,  # Label Smoothing factor
    'tta_steps': 5           # Number of TTA passes
}

# Paths
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'ready_for_fusion')
FEATURE_DIR = os.path.join(BASE_DIR, 'Extracted_Feature')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==========================================
# UTILS & DATASET
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, aud, lyr, mid, ma, ml, mm, labels):
        self.aud = aud
        self.lyr = lyr
        self.mid = mid
        self.ma = ma
        self.ml = ml
        self.mm = mm
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.aud[idx], self.lyr[idx], self.mid[idx],
            self.ma[idx], self.ml[idx], self.mm[idx],
            self.labels[idx]
        )

# ==========================================
# LOSS FUNCTIONS (Soft Targets)
# ==========================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    Used when Mixup is OFF (e.g., validation/clean training).
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        return (-weight * log_prob).sum(dim=-1).mean()

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy that accepts soft targets (probabilities).
    Used when Mixup is ON.
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# ==========================================
# MIXUP UTILS
# ==========================================
def mixup_data(a, l, m, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = a.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_a = lam * a + (1 - lam) * a[index, :]
    mixed_l = lam * l + (1 - lam) * l[index, :]
    mixed_m = lam * m + (1 - lam) * m[index, :]
    
    # We return the raw indices and lambda to construct soft labels later
    return mixed_a, mixed_l, mixed_m, y, y[index], lam

def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size(0), num_classes), off_value, device=x.device).scatter_(1, x, on_value)

def mixup_target(target, target_a, target_b, lam, num_classes, smoothing=0.0):
    """
    Creates mixed soft labels with label smoothing.
    """
    # 1. Label Smoothing
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    
    y_a = one_hot(target_a, num_classes, on_value, off_value)
    y_b = one_hot(target_b, num_classes, on_value, off_value)
    
    # 2. Mixup Interpolation
    return lam * y_a + (1 - lam) * y_b

# ==========================================
# MODEL
# ==========================================
class RobustConcatenationFusion(nn.Module):
    def __init__(self, aud_dim, lyr_dim, mid_dim, target_dim=128, num_classes=5, dropout=0.5):
        super(RobustConcatenationFusion, self).__init__()
        
        self.aud_proj = nn.Sequential(
            nn.Linear(aud_dim, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lyr_proj = nn.Sequential(
            nn.Linear(lyr_dim, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mid_proj = nn.Sequential(
            nn.Linear(mid_dim, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        concat_dim = target_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, a, l, m):
        p_a = self.aud_proj(a)
        p_l = self.lyr_proj(l)
        p_m = self.mid_proj(m)
        concat = torch.cat([p_a, p_l, p_m], dim=1)
        return self.classifier(concat)

# ==========================================
# DATA AUGMENTATION HELPERS
# ==========================================
def inject_noise(data, noise_level=0.0):
    if noise_level > 0:
        return data + torch.randn_like(data) * noise_level
    return data

def apply_modality_dropout(a, l, m, p=0.0):
    if p > 0:
        batch_size = a.size(0)
        mask_a = torch.bernoulli(torch.full((batch_size, 1), 1-p, device=a.device))
        mask_l = torch.bernoulli(torch.full((batch_size, 1), 1-p, device=a.device))
        mask_m = torch.bernoulli(torch.full((batch_size, 1), 1-p, device=a.device))
        return a * mask_a, l * mask_l, m * mask_m
    return a, l, m

# ==========================================
# VISUALIZATION UTILS
# ==========================================
def plot_learning_curves(history, output_dir):
    """
    Plots training loss and validation accuracy.
    Expects history to be a dict or list of dicts.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    if isinstance(history, list):
        for i, h in enumerate(history):
            plt.plot(h['train_loss'], label=f'Fold {i+1}')
    else:
        plt.plot(history['train_loss'], label='Train Loss')
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    if isinstance(history, list):
        for i, h in enumerate(history):
            plt.plot(h['val_acc'], label=f'Fold {i+1}')
    else:
        plt.plot(history['val_acc'], label='Val Acc')
        
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()
    print(f"Learning curve saved to {os.path.join(output_dir, 'learning_curve.png')}")

def plot_confusion_matrix(y_true, y_pred, output_dir, classes=None):
    """
    Plots confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

# ==========================================
# TRAINING LOOP
# ==========================================
def train_fold(fold, train_loader, val_loader, input_dims):
    print(f"\nFold {fold+1}")
    
    model = RobustConcatenationFusion(
        aud_dim=input_dims['audio'],
        lyr_dim=input_dims['lyrics'],
        mid_dim=input_dims['midi'],
        target_dim=CONFIG['target_dim'], 
        dropout=CONFIG['dropout'],
        num_classes=5
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], T_mult=CONFIG['T_mult'])
    
    # Loss functions
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss() # Standard for validation
    
    best_val_acc = 0
    best_model_state = None
    
    # Metrics History
    history = {
        'train_loss': [],
        'val_acc': []
    }
    
    pbar = tqdm(range(CONFIG['epochs']), desc=f"Fold {fold+1}", unit="epoch")
    
    for epoch in pbar:
        model.train()
        train_loss_accum = 0
        batch_count = 0
        
        for batch in train_loader:
            aud, lyr, mid, ma, ml, mm, labels = [b.to(device) for b in batch]
            
            # 1. Basic Augmentation
            aud = inject_noise(aud, CONFIG['noise_level'])
            lyr = inject_noise(lyr, CONFIG['noise_level'])
            mid = inject_noise(mid, CONFIG['noise_level'])
            aud, lyr, mid = apply_modality_dropout(aud, lyr, mid, CONFIG['modality_dropout'])
            
            # 2. Mixup Augmentation
            # We mix inputs and generate soft targets
            mixed_aud, mixed_lyr, mixed_mid, target_a, target_b, lam = mixup_data(
                aud, lyr, mid, labels, CONFIG['mixup_alpha']
            )
            
            soft_labels = mixup_target(labels, target_a, target_b, lam, 5, CONFIG['label_smoothing'])
            
            optimizer.zero_grad()
            outputs = model(mixed_aud, mixed_lyr, mixed_mid)
            loss = train_criterion(outputs, soft_labels)
            
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
            batch_count += 1
            
        scheduler.step()
        
        # Avg train loss for epoch
        avg_train_loss = train_loss_accum / batch_count if batch_count > 0 else 0
        history['train_loss'].append(avg_train_loss)
            
        # Validation (No Mixup, No Augmentation, Standard Loss)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                aud, lyr, mid, ma, ml, mm, labels = [b.to(device) for b in batch]
                outputs = model(aud, lyr, mid)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        history['val_acc'].append(val_acc)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            
        pbar.set_postfix({'Val Acc': f"{val_acc:.4f}", 'Best': f"{best_val_acc:.4f}", 'Loss': f"{avg_train_loss:.4f}"})
    
    return best_val_acc, best_model_state, history

# ==========================================
# TEST TIME AUGMENTATION (TTA)
# ==========================================
def predict_with_tta(model, loader, noise_level=0.02, steps=5):
    """
    Runs model inference multiple times with slight noise injection and averages results.
    """
    model.eval()
    ensemble_probs = None
    
    for _ in range(steps):
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                aud, lyr, mid, ma, ml, mm, labels = [b.to(device) for b in batch]
                
                # Inject small noise even at test time for robustness
                aud = inject_noise(aud, noise_level)
                lyr = inject_noise(lyr, noise_level)
                mid = inject_noise(mid, noise_level)
                
                outputs = model(aud, lyr, mid)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        if ensemble_probs is None:
            ensemble_probs = np.concatenate(all_probs)
        else:
            ensemble_probs += np.concatenate(all_probs)
            
    return ensemble_probs / steps

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # Load Data
        embeddings = {
            'audio': torch.load(os.path.join(OUTPUT_DIR, 'emb_audio.pt')),
            'lyrics': torch.load(os.path.join(OUTPUT_DIR, 'emb_lyrics.pt')),
            'midi': torch.load(os.path.join(OUTPUT_DIR, 'emb_midi.pt'))
        }
        song_ids = np.load(os.path.join(OUTPUT_DIR, 'song_ids.npy'), allow_pickle=True)
        
        # L2 Normalization
        print("Normalizing features...")
        embeddings['audio'] = torch.tensor(normalize(embeddings['audio'].numpy()), dtype=torch.float32)
        embeddings['lyrics'] = torch.tensor(normalize(embeddings['lyrics'].numpy()), dtype=torch.float32)
        embeddings['midi'] = torch.tensor(normalize(embeddings['midi'].numpy()), dtype=torch.float32)
        
        # Labels
        dataset_dir = '/content/drive/MyDrive/intermediate'
        if not os.path.exists(dataset_dir):
             dataset_dir = os.path.join(BASE_DIR, 'dataset')

        # We assume labels are aligned, reusing parsing logic would be verbose here, 
        # assuming user knows to ensure valid_indices logic is same as before. 
        # I'll include minimal logic to match logic.
        print("Parsing labels...")
        label_map = {}
        bat_files = [
            os.path.join(dataset_dir, 'split-by-categories-audio.bat'),
            os.path.join(dataset_dir, 'split-by-categories-lyrics.bat'),
            os.path.join(dataset_dir, 'split-by-categories-midi.bat'),
        ]
        if os.path.exists(bat_files[0]):
             with open(bat_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = re.search(r'move\s+([^\s]+)\s+\"?([^\"]+)\"?', line, flags=re.IGNORECASE)
                    if m:
                        sid = os.path.splitext(os.path.basename(m.group(1)))[0]
                        c_match = re.search(r'Cluster\s*(\d+)', m.group(2), flags=re.IGNORECASE)
                        if c_match: label_map[sid] = int(c_match.group(1)) - 1
        
        # NOTE: Just for brevity, assuming label_map is populated properly. 
        # If running in same env, should be fine.  
        # If files missing, this fails gracefully.
        
        # Re-populate alignment if empty (safeguard)
        if not label_map:
             # Try simple heuristic if bat files missing in this specific runs env
             # Assuming previous runs worked, we might've saved labels? No.
             # We must rely on bat files existing.
             pass

        aligned_labels = []
        valid_indices = []
        for idx, sid in enumerate(song_ids):
            sid_str = str(int(sid)) if str(sid).isdigit() else str(sid)
            if sid in label_map:
                aligned_labels.append(label_map[sid])
                valid_indices.append(idx)
            elif sid_str in label_map:
                aligned_labels.append(label_map[sid_str])
                valid_indices.append(idx)
                
        if len(aligned_labels) == 0:
            print("WARNING: No labels found. Ensure .bat files exist in dataset folder.")
            exit()

        y_all = np.array(aligned_labels)
        X_audio = embeddings['audio'][valid_indices]
        X_lyrics = embeddings['lyrics'][valid_indices]
        X_midi = embeddings['midi'][valid_indices]
        
        # Placeholders
        dummy_mask = torch.ones(len(y_all)) 
        
        print(f"Total samples: {len(y_all)}")
        
        # Split
        indices = np.arange(len(y_all))
        train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)
        
        # Input Dims
        input_dims = {
            'audio': X_audio.shape[1],
            'lyrics': X_lyrics.shape[1],
            'midi': X_midi.shape[1]
        }
        
        # K-Fold Training
        kfold = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
        fold_models_paths = []
        all_folds_history = []
        
        X_tv_a = X_audio[train_val_idx]
        y_tv = y_all[train_val_idx]
        
        for fold, (train_idx_local, val_idx_local) in enumerate(kfold.split(X_tv_a, y_tv)):
            train_sub_idx = train_val_idx[train_idx_local]
            val_sub_idx = train_val_idx[val_idx_local]
            
            t_ds = MultimodalDataset(
                X_audio[train_sub_idx], X_lyrics[train_sub_idx], X_midi[train_sub_idx],
                dummy_mask[train_sub_idx], dummy_mask[train_sub_idx], dummy_mask[train_sub_idx],
                y_all[train_sub_idx]
            )
            v_ds = MultimodalDataset(
                X_audio[val_sub_idx], X_lyrics[val_sub_idx], X_midi[val_sub_idx],
                dummy_mask[val_sub_idx], dummy_mask[val_sub_idx], dummy_mask[val_sub_idx],
                y_all[val_sub_idx]
            )
            
            t_loader = DataLoader(t_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
            v_loader = DataLoader(v_ds, batch_size=CONFIG['batch_size'], shuffle=False)
            
            best_acc, best_state, history = train_fold(fold, t_loader, v_loader, input_dims)
            
            model_path = os.path.join(OUTPUT_DIR, f'fold_{fold}_model_phase3.pth')
            torch.save(best_state, model_path)
            fold_models_paths.append(model_path)
            all_folds_history.append(history)
            print(f"Fold {fold+1} Acc: {best_acc:.4f}")

        # Plot Learning Curves
        print("\nPlotting learning curves...")
        plot_learning_curves(all_folds_history, OUTPUT_DIR)

        # Ensemble Inference with TTA
        print("\n=== ENSEMBLE + TTA INFERENCE ===")
        
        test_ds = MultimodalDataset(
            X_audio[test_idx], X_lyrics[test_idx], X_midi[test_idx],
            dummy_mask[test_idx], dummy_mask[test_idx], dummy_mask[test_idx],
            y_all[test_idx]
        )
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        final_probs = np.zeros((len(test_idx), 5))
        true_labels = []
        for batch in test_loader:
            true_labels.extend(batch[-1].numpy())
        
        for m_path in fold_models_paths:
            print(f"Inference using {os.path.basename(m_path)}...")
            model = RobustConcatenationFusion(
                aud_dim=input_dims['audio'],
                lyr_dim=input_dims['lyrics'],
                mid_dim=input_dims['midi'],
                target_dim=CONFIG['target_dim'], 
                dropout=CONFIG['dropout']
            ).to(device)
            model.load_state_dict(torch.load(m_path))
            
            # Predict with TTA
            fold_probs = predict_with_tta(model, test_loader, noise_level=CONFIG['noise_level'], steps=CONFIG['tta_steps'])
            final_probs += fold_probs
            
        final_probs /= CONFIG['n_splits']
        final_preds = np.argmax(final_probs, axis=1)
        
        acc = accuracy_score(true_labels, final_preds)
        print(f"Final Test Accuracy: {acc:.4f}")
        print(classification_report(true_labels, final_preds))
        
        # Plot Confusion Matrix
        print("\nPlotting confusion matrix...")
        plot_confusion_matrix(true_labels, final_preds, OUTPUT_DIR, classes=['0','1','2','3','4'])

    except Exception as e:
        import traceback
        traceback.print_exc()

