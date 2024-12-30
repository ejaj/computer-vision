import torch.nn as nn
import torchvision


class YOLOV1(nn.Module):
    """
        YOLOv1 Model with:
        1. Backbone: Pretrained ResNet-34 for feature extraction.
        2. Detection Head: 4 Conv-BatchNorm-LeakyReLU layers.
        3. Output Layer: Final layer predicts S*S*(5B+C) parameters:
            - Bounding boxes: (x, y, sqrt(w), sqrt(h), confidence).
            - Class probabilities for C classes.

        Args:
            im_size (int): Input image size.
            num_classes (int): Number of object classes (C).
            model_config (dict): Configuration for YOLO layers.

        Returns:
            torch.Tensor: Predictions of shape (Batch, S, S, 5B+C).
    """

    def __init__(self, im_size, num_classes, model_config):
        super(YOLOV1, self).__init__()
        self.im_size = im_size
        self.im_channels = model_config['im_channels']
        self.backbone_channels = model_config['backbone_channels']
        self.yolo_conv_channels = model_config['yolo_conv_channels']
        self.conv_spatial_size = model_config['conv_spatial_size']
        self.leaky_relu_slope = model_config['leaky_relu_slope']
        self.yolo_fc_hidden_dim = model_config['fc_dim']
        self.yolo_fc_dropout_prob = model_config['fc_dropout']
        self.use_conv = model_config['use_conv']
        self.S = model_config['S']
        self.B = model_config['B']
        self.C = num_classes

        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )

        # Backbone Layers
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # Detection Conv Layers
        self.conv_yolo_layers = nn.Sequential(
            nn.Conv2d(self.backbone_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope)
        )
        # Detection Layers
        if self.use_conv:
            self.fc_yolo_layers = nn.Sequential(
                nn.Conv2d(self.yolo_conv_channels, 5 * self.B + self.C, 1),
            )
        else:
            self.fc_yolo_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.conv_spatial_size * self.conv_spatial_size *
                          self.yolo_conv_channels,
                          self.yolo_fc_hidden_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.yolo_fc_dropout_prob),
                nn.Linear(self.yolo_fc_hidden_dim,
                          self.S * self.S * (5 * self.B + self.C)),
            )

    def forward(self, x):
        out = self.features(x)
        out = self.conv_yolo_layers(out)
        out = self.fc_yolo_layers(out)
        if self.use_conv:
            # Reshape conv output to Batch x S x S x (5B+C)
            out = out.permute(0, 2, 3, 1)
        return out
