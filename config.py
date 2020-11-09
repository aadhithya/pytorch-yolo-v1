"""
Model Config for YOLO v1
"""

class YoloConfig:
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        # * Define the model aechitecture.
        # * Each conv layer is a tuple (kernel_size, out_ch, stride, padding.)
        # * each conv block is a list [(conv1_params), ... , (convN_params), num_repeats]
        # * "maxpool" --> MaxPool2d with stride 2 and size 2.
        self.architecture = [
            (7, 64, 2, 3),
            "maxpool",
            (3, 192, 1, 1),
            "maxpool",
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            (1, 256, 1, 1),
            (3, 512, 1, 1),
            "maxpool",
            [(1, 256, 1, 0), (3, 1024, 1, 1), 2],
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            "maxpool",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1023, 1, 1),
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1)
        ]

        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
    