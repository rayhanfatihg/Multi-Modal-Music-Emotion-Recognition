import os
import sys
import argparse
import numpy as np
import librosa
import torch
import glob
from tqdm import tqdm

# Add the utils directory to sys.path
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from models import Cnn14
from pytorch_utils import move_data_to_device
import config

def extract_features(args):
    # Arguments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    checkpoint_path = args.checkpoint_path
    audio_dir = args.audio_dir
    output_path = args.output_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download the pre-trained model 'Cnn14_mAP=0.431.pth' and place it in the root directory or specify the correct path.")
        return

    # Model
    model = Cnn14(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    model.eval()

    # Get list of audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    features_dict = {}

    for audio_path in tqdm(audio_files):
        filename = os.path.basename(audio_path)
        
        try:
            # Load audio
            (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            
            waveform = waveform[None, :]    # (1, audio_length)
            waveform = move_data_to_device(waveform, device)

            # Forward
            with torch.no_grad():
                batch_output_dict = model(waveform, None)
            
            embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
            features_dict[filename] = embedding
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save features
    np.save(output_path, features_dict)
    print(f"Saved features to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Audio Features using PANNs CNN14')
    
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000) 
    parser.add_argument('--checkpoint_path', type=str, default='../Cnn14_mAP=0.431.pth')
    parser.add_argument('--audio_dir', type=str, default='../dataset/Audio')
    parser.add_argument('--output_path', type=str, default='../dataset/audio_features.npy')
    parser.add_argument('--cuda', action='store_true', default=True)

    args = parser.parse_args()
    
    extract_features(args)
