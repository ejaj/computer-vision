config = {
    "dataset_params": {
        "img_train_path": '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
        "ann_train_path": '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
        "img_test_path": '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
        "ann_test_path": '/home/kazi/Works/Dtu/computer-vision/recognition/fasterrcnn/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
        "num_classes": 21
    },
    "model_params": {
        "im_channels": 3,
        "aspect_ratios": [0.5, 1, 2],
        "scales": [128, 256, 512],
        "min_im_size": 600,
        "max_im_size": 1000,
        "backbone_out_channels": 512,
        "fc_inner_dim": 1024,
        "rpn_bg_threshold": 0.3,
        "rpn_fg_threshold": 0.7,
        "rpn_nms_threshold": 0.7,
        "rpn_train_prenms_topk": 12000,
        "rpn_test_prenms_topk": 6000,
        "rpn_train_topk": 2000,
        "rpn_test_topk": 300,
        "rpn_batch_size": 256,
        "rpn_pos_fraction": 0.5,
        "roi_iou_threshold": 0.5,
        "roi_low_bg_iou": 0.0,  # Increase to 0.1 for hard negatives
        "roi_pool_size": 7,
        "roi_nms_threshold": 0.3,
        "roi_topk_detections": 100,
        "roi_score_threshold": 0.05,
        "roi_batch_size": 128,
        "roi_pos_fraction": 0.25
    },
    "train_params": {
        "task_name": 'voc',
        "seed": 1111,
        "acc_steps": 1,  # Accumulation steps for larger effective batch sizes
        "num_epochs": 10,
        "lr_steps": [12, 16],
        "lr": 0.001,
        "ckpt_name": 'faster_rcnn_voc2007.pth'
    }
}
