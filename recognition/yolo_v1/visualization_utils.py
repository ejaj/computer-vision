import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

COLORS = [
    [0, 0, 0],
    [200, 0, 0],
    [0, 200, 0],
    [50, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [50, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [50, 128, 200],
    [192, 128, 200],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
]


def visualize_bbox(img, bbox, class_name, score=None, color=(255, 0, 0), thickness=2):
    """
    Draws a single bounding box on image
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    box_text = class_name + ' {:.2f}'.format(score) if score is not None else class_name
    ((text_width, text_height), _) = cv2.getTextSize(box_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=box_text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, scores=None):
    img = image.copy()
    for idx, (bbox, category_id) in enumerate(zip(bboxes, category_ids)):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, scores[idx] if scores is not None else None)
    return img


def draw_grid(img, grid_shape, color=(0, 0, 0), thickness=2):
    """
    Draws a grid on image
    """
    grid_im = np.copy(img)
    h, w, _ = grid_im.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(grid_im, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(grid_im, (0, y), (w, y), color=color, thickness=thickness)

    return grid_im


def draw_cls_grid(img, cls_idx, grid_shape):
    """
    Draws color coded grid for the entire image
    coded based on the class label
    """
    rect_im = np.copy(img)
    h, w, _ = rect_im.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    for i in range(rows):
        for j in range(cols):
            cv2.rectangle(rect_im, (int(i * dx), int(j * dy)), (int((i + 1) * dx), int((j + 1) * dy)),
                          thickness=-1,
                          color=COLORS[cls_idx[j, i].item()])
    return rect_im


def draw_cls_text(img, cls_idx, cls_idx_label, grid_shape):
    """
    Writes class text name in grid center locations
    """
    rect_im = np.copy(img)
    h, w, _ = rect_im.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    for i in range(rows):
        for j in range(cols):
            cls_label = cls_idx_label[cls_idx[j, i].item()]
            cv2.putText(rect_im,
                        cls_label[:6],
                        (int((i + 0.1) * dx), int((j + 0.5) * dy)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45,
                        color=(255, 255, 255),
                        lineType=cv2.LINE_AA)
    return rect_im


def plot_image_with_yolo_targets_and_ground_truth(image_tensor, yolo_targets, ground_truth_bboxes, S, im_size):
    """
    Plot image with YOLO targets and ground truth bounding boxes overlayed.

    Args:
        image_tensor (torch.Tensor): Image tensor of shape (3, H, W).
        yolo_targets (torch.Tensor): YOLO targets tensor of shape (S, S, 5*B + C).
        ground_truth_bboxes (torch.Tensor): Ground truth bounding boxes in [x1, y1, x2, y2] format.
        S (int): Grid size.
        im_size (int): Image size.
    """
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)  # Convert to uint8

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.axis('off')

    h, w = image_np.shape[:2]
    cell_size = im_size / S

    # Plot YOLO target boxes
    for j in range(S):
        for i in range(S):
            for b in range(2):  # Assuming B=2
                start_idx = b * 5
                conf = yolo_targets[j, i, start_idx + 4].item()
                if conf > 0:  # Only plot if confidence is greater than 0
                    xc = yolo_targets[j, i, start_idx].item() * cell_size + i * cell_size
                    yc = yolo_targets[j, i, start_idx + 1].item() * cell_size + j * cell_size
                    w_bbox = (yolo_targets[j, i, start_idx + 2].item() ** 2) * w
                    h_bbox = (yolo_targets[j, i, start_idx + 3].item() ** 2) * h

                    x1 = xc - w_bbox / 2
                    y1 = yc - h_bbox / 2
                    x2 = xc + w_bbox / 2
                    y2 = yc + h_bbox / 2

                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            edgecolor='red', facecolor='none', linewidth=2, label="YOLO Targets"
                        )
                    )

    # Plot ground truth boxes
    for bbox in ground_truth_bboxes:
        x1, y1, x2, y2 = bbox * torch.Tensor([w, h, w, h])  # Rescale to original size
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='blue', facecolor='none', linewidth=2, linestyle='--', label="Ground Truth"
            )
        )

    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', lw=2, label='YOLO Targets'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Ground Truth')
    ], loc='upper right')
    plt.show()
