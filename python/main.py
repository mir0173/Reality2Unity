"""Week 1 baseline from the 2024 appendix, through 3D visualization.

Run from the repository root: python python/main.py [image_path]
GeoTIFF export is reserved for the next stage.
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from depth_estimation.midas import load_midas_model, estimate_depth
from terrain_generation.legacy_converter import normalize_depth, convert_to_top_view


def visualize_depth(image_path, depth_map):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Normalized Height Map")
    plt.imshow(depth_map, cmap="viridis")
    plt.colorbar(label="Height (0 to 100)")
    plt.axis("off")
    plt.show()


def visualize_3d(Z):
    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(x, y)
    X -= X.min()
    Y -= Y.min()
    Z -= Z.min()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='none')
    ax.set_title('3D Height Map Visualization')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Height')
    plt.show()


def main(image_path):
    midas, transform, device = load_midas_model()
    depth_map = estimate_depth(image_path, midas, transform, device)
    depth_map[depth_map < 0] = 0
    height_map = normalize_depth(depth_map, 100)
    height_map = np.max(height_map) - height_map
    visualize_depth(image_path, height_map)
    top_view_depth_map = normalize_depth(height_map, depth_map.shape[0])
    top_view_height_map = normalize_depth(convert_to_top_view(top_view_depth_map), 100)
    visualize_depth(image_path, top_view_height_map)
    visualize_3d(top_view_height_map)


if __name__ == "__main__":
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else 'image.jpg'
    main(test_image_path)
