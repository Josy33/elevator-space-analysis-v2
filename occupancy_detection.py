"""
Occupancy Detection Model for Elevator Space Analysis
This script trains a model to detect occupancy and calculate available space in elevators
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
print("=== DEBUG INFO ===")
print(f"Current directory: {os.getcwd()}")
print(f"Data path exists: {os.path.exists('data/output/depth_estimation')}")
print(f"Empty dir exists: {os.path.exists('data/output/depth_estimation/empty')}")
print(f"Occupied dir exists: {os.path.exists('data/output/depth_estimation/occupied')}")

# Check for actual files
import glob
empty_files = glob.glob("data/output/depth_estimation/empty/*_depth.jpg")
occupied_files = glob.glob("data/output/depth_estimation/occupied/*_depth.jpg")
print(f"Empty depth maps found: {len(empty_files)}")
print(f"Occupied depth maps found: {len(occupied_files)}")
print("==================")

print("===== ELEVATOR OCCUPANCY DETECTION =====")

# Set paths
data_path = "data/output/depth_estimation"
output_path = "data/output/occupancy"
os.makedirs(output_path, exist_ok=True)

# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class for depth maps
class DepthMapDataset(Dataset):
    def __init__(self, depth_maps, labels, transform=None):
        self.depth_maps = depth_maps
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.depth_maps)
    
    def __getitem__(self, idx):
        # Load depth map
        depth_map = cv2.imread(self.depth_maps[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert to 3 channels (for compatibility with pre-trained models)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms if any
        if self.transform:
            depth_map = self.transform(depth_map)
        
        return depth_map, self.labels[idx]

# Model class for occupancy detection
class OccupancyDetector(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(OccupancyDetector, self).__init__()
        
        # Load pretrained model
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add our own classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SpaceCalculator:
    """Utility class to calculate available space from depth maps"""
    
    @staticmethod
    def calculate_space(depth_map, threshold=0.5):
        """
        Calculate available space percentage in an elevator based on depth map.
        
        Args:
            depth_map: Grayscale depth map (0-255)
            threshold: Depth threshold (0-1) for considering space as occupied
            
        Returns:
            float: Percentage of available space (0-100)
        """
        # Normalize depth map to 0-1
        depth_norm = depth_map.astype(float) / 255.0
        
        # Calculate occupied space (lower depth values = closer to camera = occupied)
        # We invert the depth map so higher values = closer to camera
        occupied_mask = (1.0 - depth_norm) > threshold
        
        # Calculate percentage of available space
        total_pixels = depth_map.size
        occupied_pixels = np.sum(occupied_mask)
        available_space = 100.0 * (1.0 - occupied_pixels / total_pixels)
        
        return available_space
    
    @staticmethod
    def create_visualization(depth_map, threshold=0.5):
        """
        Create a visualization of occupied vs. available space.
        
        Args:
            depth_map: Grayscale depth map (0-255)
            threshold: Depth threshold (0-1) for considering space as occupied
            
        Returns:
            numpy.ndarray: RGB visualization image
        """
        # Normalize depth map to 0-1
        depth_norm = depth_map.astype(float) / 255.0
        
        # Create color visualization
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        # Calculate occupied space
        occupied_mask = ((1.0 - depth_norm) > threshold) * 255
        occupied_mask = occupied_mask.astype(np.uint8)
        
        # Create a color mask for occupied areas
        occupied_color = np.zeros_like(depth_color)
        occupied_color[occupied_mask > 0] = [0, 0, 255]  # Red color for occupied space
        
        # Blend depth map and occupancy mask
        alpha = 0.4
        visualization = cv2.addWeighted(depth_color, 1.0, occupied_color, alpha, 0)
        
        # Calculate available space
        available_space = 100.0 * (1.0 - np.sum(occupied_mask > 0) / depth_map.size)
        
        # Add text with available space percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, f"Available: {available_space:.1f}%", 
                  (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Determine occupancy status based on available space
        status = "SPACE AVAILABLE"
        status_color = (0, 255, 0)  # Green
        
        if available_space < 30:
            status = "ELEVATOR FULL"
            status_color = (0, 0, 255)  # Red
        elif available_space < 60:
            status = "LIMITED SPACE"
            status_color = (0, 165, 255)  # Orange
        
        cv2.putText(visualization, status, 
                  (10, visualization.shape[0] - 20), font, 0.7, status_color, 2, cv2.LINE_AA)
        
        return visualization

def train_occupancy_model():
    """Train the occupancy detection model using depth maps"""
    print("\nPreparing dataset for occupancy detection...")
    
    # Collect depth maps and labels
    empty_depth_maps = glob.glob(os.path.join(data_path, "empty", "*_depth.jpg"))
    occupied_depth_maps = glob.glob(os.path.join(data_path, "occupied", "*_depth.jpg"))
    
    print(f"Found {len(empty_depth_maps)} empty depth maps")
    print(f"Found {len(occupied_depth_maps)} occupied depth maps")
    
    # Create dataset
    all_depth_maps = empty_depth_maps + occupied_depth_maps
    labels = [0] * len(empty_depth_maps) + [1] * len(occupied_depth_maps)
    
    # Split dataset into train and validation sets
    train_maps, val_maps, train_labels, val_labels = train_test_split(
        all_depth_maps, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_maps)} depth maps")
    print(f"Validation set: {len(val_maps)} depth maps")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DepthMapDataset(train_maps, train_labels, transform=train_transform)
    val_dataset = DepthMapDataset(val_maps, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = OccupancyDetector(backbone='resnet18', pretrained=True)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.float().view(-1, 1).to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets for metrics
                preds = (outputs > 0.5).float().cpu().numpy()
                true = targets.cpu().numpy()
                train_preds.extend(preds)
                train_true.extend(true)
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_true, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for inputs, targets in pbar:
                    inputs = inputs.to(device)
                    targets = targets.float().view(-1, 1).to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Store predictions and targets for metrics
                    preds = (outputs > 0.5).float().cpu().numpy()
                    true = targets.cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(true)
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item()})
        
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}: ")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_path, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to {model_path}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pth")))
    
    # Evaluate on validation set
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            preds = (outputs > 0.5).float().cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(targets.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(val_true, val_preds)
    precision = precision_score(val_true, val_preds)
    recall = recall_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds)
    conf_matrix = confusion_matrix(val_true, val_preds)
    
    print("\nFinal Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("  Confusion Matrix:")
    print(f"    TN: {conf_matrix[0][0]}, FP: {conf_matrix[0][1]}")
    print(f"    FN: {conf_matrix[1][0]}, TP: {conf_matrix[1][1]}")
    
    return model

def test_on_new_images(model, test_dir):
    """Test the model on new images from the test directory"""
    print("\nTesting model on new images...")
    
    # Define transform
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get test images
    test_depth_maps = glob.glob(os.path.join(test_dir, "*_depth.jpg"))
    print(f"Found {len(test_depth_maps)} test depth maps")
    
    if not test_depth_maps:
        print("No test images found")
        return
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_path, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each test image
    results = []
    
    with tqdm(test_depth_maps, desc="Processing test images") as pbar:
        for depth_path in pbar:
            # Get file name
            filename = os.path.basename(depth_path)
            name, _ = os.path.splitext(filename)
            name = name.replace("_depth", "")
            
            # Load depth map
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            
            # Prepare for model
            rgb_depth = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
            model_input = test_transform(rgb_depth).unsqueeze(0).to(device)
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                prediction = model(model_input)
                prob = prediction.item()
                occupied = prob > 0.5
            
            # Calculate available space
            available_space = SpaceCalculator.calculate_space(depth_map)
            
            # Create visualization
            visualization = SpaceCalculator.create_visualization(depth_map)
            
            # Add model prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            status = "OCCUPIED" if occupied else "EMPTY"
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.putText(visualization, f"Prediction: {status} ({prob:.2f})", 
                       (10, 60), font, 0.7, color, 2, cv2.LINE_AA)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f"{name}_analyzed.jpg")
            cv2.imwrite(vis_path, visualization)
            
            # Store result
            results.append({
                "filename": filename,
                "occupied": occupied,
                "probability": prob,
                "available_space": available_space
            })
            
            pbar.set_postfix({"occupied": occupied, "space": f"{available_space:.1f}%"})
    
    print(f"\nProcessed {len(results)} test images")
    print(f"Visualizations saved to {vis_dir}")
    
    return results

def main():
    """Main function to run the occupancy detection pipeline"""
    # Train occupancy detection model
    model = train_occupancy_model()
    
    # Test on new images
    test_dir = os.path.join(data_path, "test")
    if os.path.exists(test_dir):
        results = test_on_new_images(model, test_dir)
    
    print("\nOccupancy detection complete!")
    print(f"Model and visualizations saved to {output_path}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Review the visualizations in the output directory")
    print("2. Run space_analyzer.py to analyze real-time elevator feeds")
    print("3. Integrate with the elevator control system")

if __name__ == "__main__":
    main()