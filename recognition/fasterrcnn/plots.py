import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_images_in_row(dataset, idx2label, num_images=3):
    """
    Plot a row of images with bounding boxes and labels.
    
    Args:
        dataset (Dataset): The dataset object (VOCDataset instance).
        idx2label (dict): Mapping from class index to class label.
        num_images (int): Number of images to plot in a row.
    """
    # Create a figure with 1 row and `num_images` columns
    fig, axes = plt.subplots(1, num_images, figsize=(20, 8))

    for i, ax in enumerate(axes):
        # Fetch data using __getitem__
        img_tensor, targets, filename = dataset[i]  
        
        # Convert tensor to numpy array for plotting
        img = img_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

        # Display the image
        ax.imshow(img)
        ax.set_title(f"Image: {filename.split('/')[-1]}")
        ax.axis('off')

        # Plot bounding boxes
        for bbox, label_idx in zip(targets['bboxes'], targets['labels']):
            x1, y1, x2, y2 = bbox.tolist()
            label = idx2label[label_idx.item()]
            # Create a rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            # Add label
            ax.text(
                x1, y1 - 10, label, color='red', fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

    plt.tight_layout()
    plt.show()
