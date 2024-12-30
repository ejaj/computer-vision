from tqdm import tqdm
import glob
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import random
import torch
import torchvision
from plots import plot_images_in_row

def load_images_and_anns(im_dir, ann_dir, label2idx):
    """
    Load image metadata and annotations for object detection.

    Args:
        im_dir (str): Directory containing image files.
        ann_dir (str): Directory containing annotation XML files.
        label2idx (dict): Mapping of class labels to their corresponding indices.

    Returns:
        list[dict]: List of dictionaries containing:
            - 'img_id' (str): Image ID.
            - 'filename' (str): Path to the image file.
            - 'width' (int): Image width.
            - 'height' (int): Image height.
            - 'detections' (list[dict]): List of detections with:
                - 'label' (int): Class label index.
                - 'bbox' (list[int]): Bounding box [xmin, ymin, xmax, ymax].

    Notes:
        - Annotation files must be in PASCAL VOC XML format.
        - Bounding box coordinates are zero-indexed.
    """
    img_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        img_info = {}
        img_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        img_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(img_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        img_info['width'] = width
        img_info['height'] = height

        detections = []

        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        img_info['detections'] = detections
        img_infos.append(img_info)
    print('Total {} images found'.format(len(img_infos)))
    return img_infos


class VOCDataset(Dataset):
    def __init__(self, split, img_dir, ann_dir):
        self.split = split
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        classes = sorted(classes)
        classes = ['background'] + classes

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        
        self.images_info = load_images_and_anns(img_dir, ann_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = Image.open(img_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in img_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in img_info['detections']])

        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = img_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return img_tensor, targets, img_info['filename']


# if __name__ == "__main__":
    
#     im_train_path = '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
#     ann_train_path = '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'

#     # Create the dataset instance for training
#     train_dataset = VOCDataset(split='train', img_dir=im_train_path, ann_dir=ann_train_path)
  

#     # Test the dataset length
#     print(f"Number of images in training dataset: {len(train_dataset)}")

#     plot_images_in_row(train_dataset, train_dataset.idx2label)
