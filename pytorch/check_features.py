import numpy as np
import os

file_path = '../dataset/audio_features.npy'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Load the file
    # allow_pickle=True is required because we saved a dictionary
    data = np.load(file_path, allow_pickle=True)
    
    # np.save wraps the dictionary in a 0-d array, so we use .item() to retrieve it
    features_dict = data.item()
    
    print(f"Successfully loaded features from {file_path}")
    print(f"Total number of audio files processed: {len(features_dict)}")
    
    if len(features_dict) > 0:
        # Get the first key and value to check shape
        first_filename = list(features_dict.keys())[0]
        first_embedding = features_dict[first_filename]
        
        print(f"\nExample entry:")
        print(f"Filename: {first_filename}")
        print(f"Embedding shape: {first_embedding.shape}")
        
        # Verify expected shape (should be 2048 for Cnn14)
        if first_embedding.shape == (2048,):
            print("\nShape verification: PASSED (Expected (2048,))")
        else:
            print(f"\nShape verification: WARNING (Expected (2048,), got {first_embedding.shape})")
    else:
        print("The dictionary is empty.")
