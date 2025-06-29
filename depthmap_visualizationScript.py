# Quick visualization script
import cv2
import os
import matplotlib.pyplot as plt

categories = ['empty', 'partially_occupied', 'full']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cat in enumerate(categories):
    # Get first depth map from each category
    depth_path = f"data/output/depth_estimation/{cat}/"
    first_depth = os.listdir(depth_path)[0]
    if first_depth.endswith('_depth.jpg'):
        img = cv2.imread(os.path.join(depth_path, first_depth), 0)
        axes[i].imshow(img, cmap='inferno')
        axes[i].set_title(f'{cat} - Sample Depth Map')
        axes[i].axis('off')

plt.tight_layout()
plt.show()