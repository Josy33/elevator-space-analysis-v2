# test_midas_simple.py
import os
import sys
import torch
import cv2
import numpy as np

# Add MiDaS path
midas_path = r"C:\Users\ephra\OneDrive\Documents\Desktop\MiDaS-master\MiDaS-master"
sys.path.append(midas_path)
print(f"MiDaS path: {midas_path}")
print(f"Path exists: {os.path.exists(midas_path)}")

# Let's check the model weights directory
weights_dir = os.path.join(midas_path, "weights")
if not os.path.exists(weights_dir):
    print(f"Creating weights directory at: {weights_dir}")
    os.makedirs(weights_dir, exist_ok=True)

# Check for model weights file
small_weights = os.path.join(weights_dir, "midas_v21_small-70d6b9c8.pt")
if not os.path.exists(small_weights):
    print(f"Model weights not found at: {small_weights}")
    print("Downloading MiDaS small model weights...")
    
    import urllib.request
    url = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt"
    try:
        urllib.request.urlretrieve(url, small_weights)
        print(f"Downloaded weights to: {small_weights}")
    except Exception as e:
        print(f"Error downloading weights: {e}")

# Try to import modules
try:
    from midas.midas_net_custom import MidasNet_small
    print("Successfully imported MidasNet_small")
except ImportError as e:
    print(f"Error importing MidasNet_small: {e}")
    sys.exit(1)

# Try to load the model
try:
    print("Attempting to load model...")
    model = MidasNet_small(
        small_weights,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={'expand': True}
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Try to process a simple image
    print("Creating test image...")
    test_img = np.zeros((320, 240, 3), dtype=np.uint8)
    test_img[:, :, 0] = 255  # Blue test image
    
    # Convert to RGB and normalize
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) / 255.0
    
    # Resize to model input size
    test_img = cv2.resize(test_img, (256, 256))
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    test_img = (torch.from_numpy(test_img).permute(2, 0, 1) - mean[:, None, None]) / std[:, None, None]
    
    # Add batch dimension
    test_img = test_img.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        prediction = model(test_img)
        print("Model inference successful!")
    
except Exception as e:
    print(f"Error with model: {e}")
    import traceback
    traceback.print_exc()
    