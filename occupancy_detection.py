"""
Enhanced Multi-Class Occupancy Detection with Geometric Features
This script trains a model to classify elevator occupancy into: Empty, Partially Occupied, or Full
Uses both depth map images and geometric features for improved accuracy
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

print("===== ENHANCED ELEVATOR OCCUPANCY DETECTION (3-CLASS) =====")

# Set paths
data_path = "data/output/depth_estimation"
output_path = "data/output/occupancy"
os.makedirs(output_path, exist_ok=True)

# Define classes
CLASS_NAMES = ['empty', 'partially_occupied', 'full']
CLASS_LABELS = {'empty': 0, 'partially_occupied': 1, 'full': 2}

# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_geometric_features(depth_meta_path):
    """Extract and normalize the most discriminative features from metadata"""
    try:
        with open(depth_meta_path, 'r') as f:
            metadata = json.load(f)
        
        stats = metadata['depth_stats']
        
        # Extract the features that show good differentiation based on your analysis
        features = np.array([
            stats['edge_density'],           # Good differentiation: 3.37 → 4.60 → 5.11
            stats['near_pixel_ratio'],       # Excellent: 0.068 → 0.140 → 0.170
            stats['center_periphery_ratio'], # Good: 0.67 → 0.89 → 0.99
            stats.get('far_pixel_ratio', stats.get('floor_visibility_ratio', 0)),  # Floor visibility
            stats.get('edge_std', 0),        # Edge variation
            stats.get('vertical_variation', 0),   # Vertical complexity
            stats.get('horizontal_variation', 0), # Horizontal complexity
            stats.get('texture_complexity', 0),   # Texture measure
        ], dtype=np.float32)
        
        return features
    except Exception as e:
        # Return zeros if metadata reading fails
        print(f"Warning: Could not read metadata from {depth_meta_path}: {e}")
        return np.zeros(8, dtype=np.float32)

# Dataset class for depth maps with geometric features
class EnhancedDepthMapDataset(Dataset):
    def __init__(self, depth_maps, labels, transform=None, use_features=True):
        self.depth_maps = depth_maps
        self.labels = labels
        self.transform = transform
        self.use_features = use_features
        
        # Extract all geometric features upfront for normalization
        if self.use_features:
            print("Extracting geometric features...")
            self.geometric_features = []
            for depth_path in tqdm(depth_maps, desc="Loading features"):
                meta_path = depth_path.replace('_depth.jpg', '_depth_meta.json')
                features = extract_geometric_features(meta_path)
                self.geometric_features.append(features)
            
            # Fit scaler on training data features
            self.scaler = StandardScaler()
            self.geometric_features = self.scaler.fit_transform(self.geometric_features).astype(np.float32)
            print(f"Loaded features shape: {self.geometric_features.shape}")
        
    def __len__(self):
        return len(self.depth_maps)
    
    def __getitem__(self, idx):
        # Load depth map
        depth_map = cv2.imread(self.depth_maps[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert to 3 channels for compatibility with pre-trained models
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            depth_map = self.transform(depth_map)
        
        if self.use_features:
            # Convert features to float32 tensor
            features = torch.FloatTensor(self.geometric_features[idx])
            return depth_map, self.labels[idx], features
        else:
            return depth_map, self.labels[idx]

class OccupancyDetectorWithFeatures(nn.Module):
    """ResNet18-based model with geometric feature fusion for 3-class occupancy detection"""
    def __init__(self, num_classes=3, num_features=8, use_features=True):
        super().__init__()
        self.use_features = use_features
        
        # Load pre-trained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze early layers (optional - can improve training)
        for param in list(self.base_model.parameters())[:-20]:
            param.requires_grad = False
        
        # Get the number of features from ResNet
        num_visual_features = self.base_model.fc.in_features
        
        # Remove the final FC layer
        self.base_model.fc = nn.Identity()
        
        if self.use_features:
            # Feature processing branch
            self.feature_branch = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU()
            )
            
            # Combined classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_visual_features + 16, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        else:
            # Visual-only classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_visual_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x, features=None):
        # Extract visual features
        visual_features = self.base_model(x)
        
        if self.use_features and features is not None:
            # Process geometric features
            geom_features = self.feature_branch(features)
            # Concatenate visual and geometric features
            combined = torch.cat([visual_features, geom_features], dim=1)
            return self.classifier(combined)
        else:
            return self.classifier(visual_features)

# Keep the SpaceCalculator class from before
class SpaceCalculator:
    """Enhanced utility class to calculate available space from depth maps"""
    
    @staticmethod
    def calculate_space_percentage(depth_map):
        """Calculate available space percentage using depth analysis."""
        depth_norm = depth_map.astype(float) / 255.0
        
        # Use adaptive thresholds based on your data
        floor_threshold = 0.3
        person_threshold = 0.7
        
        floor_mask = depth_norm < floor_threshold
        occupied_mask = depth_norm > person_threshold
        
        total_floor_pixels = np.sum(floor_mask)
        occupied_floor_pixels = np.sum(floor_mask & occupied_mask)
        
        if total_floor_pixels > 0:
            available_space = 100.0 * (1.0 - occupied_floor_pixels / total_floor_pixels)
        else:
            available_space = 100.0 * (1.0 - np.sum(occupied_mask) / depth_map.size)
        
        return available_space
    
    @staticmethod
    def classify_occupancy(available_space):
        """Classify occupancy based on available space percentage."""
        if available_space > 70:
            return 'empty', 0
        elif available_space > 30:
            return 'partially_occupied', 1
        else:
            return 'full', 2
    
    @staticmethod
    def create_visualization(depth_map, prediction=None, confidence=None, geometric_info=None):
        """Create an enhanced visualization of occupancy analysis."""
        # Create base visualization
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        # Calculate space percentage
        available_space = SpaceCalculator.calculate_space_percentage(depth_map)
        space_class, _ = SpaceCalculator.classify_occupancy(available_space)
        
        # Create overlay
        depth_norm = depth_map.astype(float) / 255.0
        occupied_mask = (depth_norm > 0.7).astype(np.uint8) * 255
        
        # Color code based on occupancy
        overlay = np.zeros_like(depth_color)
        if space_class == 'empty':
            overlay[:, :] = [0, 255, 0]  # Green
        elif space_class == 'partially_occupied':
            overlay[:, :] = [0, 165, 255]  # Orange
        else:
            overlay[:, :] = [0, 0, 255]  # Red
        
        # Apply overlay
        mask_3ch = cv2.cvtColor(occupied_mask, cv2.COLOR_GRAY2BGR)
        # Create weighted overlay
        overlay_weighted = overlay * (mask_3ch / 255.0)
        visualization = cv2.addWeighted(depth_color, 0.7, overlay_weighted.astype(np.uint8), 0.3, 0)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Available space
        cv2.putText(visualization, f"Floor Space: {available_space:.1f}%", 
                   (10, y_offset), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30
        
        # Add geometric feature info if provided
        if geometric_info is not None:
            cv2.putText(visualization, f"Edge Density: {geometric_info[0]:.2f}", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
            cv2.putText(visualization, f"Near Objects: {geometric_info[1]:.2f}", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
        
        # Model prediction (if provided)
        if prediction is not None and confidence is not None:
            pred_class = CLASS_NAMES[prediction]
            pred_conf = confidence[prediction]
            cv2.putText(visualization, f"Model: {pred_class.upper()} ({pred_conf:.2f})", 
                       (10, y_offset), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Status bar at bottom
        status_height = 40
        status_y = visualization.shape[0] - status_height
        
        # Determine final status
        if prediction is not None:
            final_class = CLASS_NAMES[prediction]
        else:
            final_class = space_class
        
        # Color based on status
        if final_class == 'empty':
            status_color = (0, 255, 0)
            status_text = "ELEVATOR AVAILABLE"
        elif final_class == 'partially_occupied':
            status_color = (0, 165, 255)
            status_text = "LIMITED SPACE"
        else:
            status_color = (0, 0, 255)
            status_text = "ELEVATOR FULL"
        
        # Draw status bar
        cv2.rectangle(visualization, (0, status_y), 
                     (visualization.shape[1], visualization.shape[0]), 
                     status_color, -1)
        cv2.putText(visualization, status_text, 
                   (10, visualization.shape[0] - 10), 
                   font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return visualization

def train_occupancy_model():
    """Train the enhanced 3-class occupancy detection model using depth maps and geometric features"""
    # Check if already trained
    model_path = os.path.join(output_path, "best_model_enhanced.pth")
    if os.path.exists(model_path):
        print("\nFound existing trained model!")
        response = input("Do you want to (r)etrain or (s)kip to testing? [s]: ").lower()
        if response != 'r':
            print("Loading existing model...")
            model = OccupancyDetectorWithFeatures(num_classes=3, num_features=8, use_features=True)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print(f"Loaded model with {checkpoint['best_acc']:.2%} validation accuracy")
            return model
    
    # If no model or retraining requested, proceed with training
    print("\nPreparing dataset for enhanced 3-class occupancy detection...")
    
    # Collect depth maps and labels
    all_depth_maps = []
    all_labels = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    for class_name, label in CLASS_LABELS.items():
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            depth_maps = glob.glob(os.path.join(class_path, "*_depth.jpg"))
            all_depth_maps.extend(depth_maps)
            all_labels.extend([label] * len(depth_maps))
            class_counts[class_name] = len(depth_maps)
            print(f"Found {len(depth_maps)} {class_name} depth maps")
    
    if not all_depth_maps:
        print("ERROR: No depth maps found! Please run depth estimation first.")
        return None
    
    print(f"\nTotal depth maps: {len(all_depth_maps)}")
    print(f"Class distribution: {class_counts}")
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(all_labels), 
                                       y=all_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
    
    # Split dataset
    train_maps, val_maps, train_labels, val_labels = train_test_split(
        all_depth_maps, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"\nTraining set: {len(train_maps)} depth maps")
    print(f"Validation set: {len(val_maps)} depth maps")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(5),  # Small rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with geometric features
    print("\nCreating enhanced datasets with geometric features...")
    train_dataset = EnhancedDepthMapDataset(train_maps, train_labels, 
                                          transform=train_transform, use_features=True)
    val_dataset = EnhancedDepthMapDataset(val_maps, val_labels, 
                                        transform=val_transform, use_features=True)
    
    # Save the scaler for later use
    scaler_path = os.path.join(output_path, "feature_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    print(f"Saved feature scaler to {scaler_path}")
    
    # Create data loaders
    batch_size = 16 if device.type == 'cpu' else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model with feature fusion
    print("\nInitializing enhanced 3-class occupancy model with feature fusion...")
    model = OccupancyDetectorWithFeatures(num_classes=3, num_features=8, use_features=True)
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    
    # Training setup
    num_epochs = 40  # Increased epochs
    best_val_acc = 0
    patience_counter = 0
    max_patience = 12  # Increased patience
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting training with visual + geometric features...")
    print("Expected accuracy: 70-85% (realistic with current feature differentiation)")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for batch in pbar:
                if len(batch) == 3:  # With features
                    inputs, targets, features = batch
                    features = features.to(device)
                else:  # Without features
                    inputs, targets = batch
                    features = None
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs, features)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(targets.cpu().numpy())
                
                pbar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_true, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        val_probs = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for batch in pbar:
                    if len(batch) == 3:  # With features
                        inputs, targets, features = batch
                        features = features.to(device)
                    else:  # Without features
                        inputs, targets = batch
                        features = None
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs, features)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(targets.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    
                    pbar.set_postfix({"loss": loss.item()})
        
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(output_path, "best_model_enhanced.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'class_weights': class_weights,
                'use_features': True,
                'num_features': 8,
                'best_acc': best_val_acc
            }, model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(os.path.join(output_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, cls=NumpyEncoder)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_path, "best_model_enhanced.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions for final evaluation
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                inputs, targets, features = batch
                features = features.to(device)
            else:
                inputs, targets = batch
                features = None
            
            inputs = inputs.to(device)
            outputs = model(inputs, features)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(targets.cpu().numpy())
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(val_true, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (Best Acc: {best_val_acc:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrix_enhanced.png'))
    plt.close()
    
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Confusion matrix saved to {output_path}/confusion_matrix_enhanced.png")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_history.png'))
    plt.close()
    
    return model

def test_on_new_images(model, test_dir):
    """Test the model on new images from the test directory with checkpoint support"""
    print("\nTesting enhanced model on new images...")
    
    # Load feature scaler
    scaler_path = os.path.join(output_path, "feature_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
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
    
    # Create output directory
    vis_dir = os.path.join(output_path, "test_visualizations_enhanced")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Check for existing results (checkpoint functionality)
    results_path = os.path.join(output_path, "test_results_enhanced.json")
    existing_results = []
    processed_files = set()
    
    if os.path.exists(results_path):
        print("Found existing test results, loading checkpoint...")
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            processed_files = {r['filename'] for r in existing_results}
            print(f"Already processed: {len(processed_files)} images")
        except:
            print("Could not load existing results, starting fresh")
            existing_results = []
    
    # Filter out already processed images
    remaining_depth_maps = []
    for depth_path in test_depth_maps:
        filename = os.path.basename(depth_path)
        if filename not in processed_files:
            remaining_depth_maps.append(depth_path)
    
    print(f"Remaining to process: {len(remaining_depth_maps)} images")
    
    if not remaining_depth_maps:
        print("All test images already processed!")
        return existing_results
    
    # Process remaining images (limit to 100 per run for memory efficiency)
    batch_size = min(100, len(remaining_depth_maps))
    results = existing_results.copy()
    
    print(f"Processing batch of {batch_size} images...")
    
    with tqdm(remaining_depth_maps[:batch_size], desc="Processing test images") as pbar:
        for depth_path in pbar:
            filename = os.path.basename(depth_path)
            name = filename.replace("_depth.jpg", "")
            
            # Load depth map
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract geometric features
            meta_path = depth_path.replace('_depth.jpg', '_depth_meta.json')
            raw_features = extract_geometric_features(meta_path)
            features = scaler.transform([raw_features])[0].astype(np.float32)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            # Prepare image for model
            rgb_depth = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
            model_input = test_transform(rgb_depth).unsqueeze(0).to(device)
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                outputs = model(model_input, features_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
            pred_class = predicted.item()
            pred_name = CLASS_NAMES[pred_class]
            confidences = probs[0].cpu().numpy()
            
            # Calculate space percentage
            available_space = SpaceCalculator.calculate_space_percentage(depth_map)
            
            # Create visualization with geometric info
            visualization = SpaceCalculator.create_visualization(
                depth_map, pred_class, confidences, raw_features
            )
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f"{name}_analyzed.jpg")
            cv2.imwrite(vis_path, visualization)
            
            # Store result with explicit type conversion
            result = {
                "filename": str(filename),
                "predicted_class": str(pred_name),
                "confidence": float(confidence.item() if hasattr(confidence, 'item') else confidence),
                "all_confidences": {
                    str(cls): float(conf) for cls, conf in zip(CLASS_NAMES, confidences)
                },
                "available_space": float(available_space),
                "edge_density": float(raw_features[0]),
                "near_pixel_ratio": float(raw_features[1])
            }
            results.append(result)
            
            pbar.set_postfix({
                "class": pred_name, 
                "conf": f"{confidence.item():.2f}",
                "space": f"{available_space:.1f}%"
            })
    
    # Save results (incremental save for checkpoint)
    results_path = os.path.join(output_path, "test_results_enhanced.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nProcessed {batch_size} images in this batch")
    print(f"Total processed so far: {len(results)} images")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    for r in results:
        class_counts[r['predicted_class']] += 1
    
    print("\nTest Set Predictions Summary (all processed images):")
    for cls, count in class_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    
    if len(remaining_depth_maps) > batch_size:
        print(f"\n{len(remaining_depth_maps) - batch_size} images remaining. Run again to continue processing.")
    else:
        print("\nAll test images processed!")
    
    return results

def main():
    """Main function to run the enhanced 3-class occupancy detection pipeline"""
    # Train model
    model = train_occupancy_model()
    
    if model is None:
        print("Training failed. Exiting.")
        return
    
    # Test on new images if test directory exists
    test_dir = os.path.join(data_path, "test")
    if os.path.exists(test_dir):
        results = test_on_new_images(model, test_dir)
    else:
        print(f"\nNo test directory found at {test_dir}")
    
    print("\n" + "="*60)
    print("ENHANCED 3-CLASS OCCUPANCY DETECTION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {output_path}")
    print("\nWhat to expect:")
    print("- Accuracy: 70-85% (realistic with current features)")
    print("- Best performance on 'empty' and 'full' classes")
    print("- Some confusion between 'partially_occupied' and 'full'")
    print("\nNext steps:")
    print("1. Review confusion matrix and test visualizations")
    print("2. Look for patterns in misclassifications")
    print("3. Consider collecting more diverse training data")
    print("4. Deploy for real-world testing and feedback")

if __name__ == "__main__":
    main()