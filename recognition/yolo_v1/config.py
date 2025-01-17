# Config Dictionary
config = {
    "dataset_params": {
        "train_im_sets": [
            "/home/kazi/Works/Dtu/computer-vision/recognition/yolo_v1/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007",
            "/home/kazi/Works/Dtu/computer-vision/recognition/yolo_v1/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
        ],
        "test_im_sets": [
            "/home/kazi/Works/Dtu/computer-vision/recognition/yolo_v1/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007"
        ],
        "num_classes": 20,
        "im_size": 448
    },
    "model_params": {
        "im_channels": 3,
        "backbone_channels": 512,
        "conv_spatial_size": 7,
        "yolo_conv_channels": 1024,
        "leaky_relu_slope": 0.1,
        "fc_dim": 4096,
        "fc_dropout": 0.5,
        "S": 7,
        "B": 2,
        "use_sigmoid": True,
        "use_conv": True
    },
    "train_params": {
        "task_name": "voc",
        "seed": 1111,
        "acc_steps": 1,  # Gradient accumulation steps
        "log_steps": 100,
        "num_epochs": 5,
        "batch_size": 10,
        "lr_steps": [50, 75, 100, 125],
        "lr": 0.001,
        "infer_conf_threshold": 0.2,
        "eval_conf_threshold": 0.001,
        "nms_threshold": 0.5,
        "ckpt_name": "yolo_voc2007.pth"
    }
}
