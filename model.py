import torch
import torch.nn as nn
from torch.nn.modules import linear
from config import YoloConfig

from pytorch_model_summary import summary

class ConvLayer(nn.Module):
    """
    ConvLayer Convolutional Layer. Performs convolution, batchniorm and leaky relu.
    """
    def __init__(self, in_channels:int, out_channels:int, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv_block(x)


class YOLOv1(nn.Module):
    def __init__(self, cfg:YoloConfig):
        super().__init__()
        self.cfg = cfg
        # * create backbone network darknet as defined in the paper.
        self.darknet = self.__create_conv_layers(self.cfg)
        # * create the final FC layers
        self.linear_block = self.__create_linear_block(self.cfg)

    def forward(self, x):
        # * get darknet feature maps
        x = self.darknet(x)
        # * flatten feature map.
        x = torch.flatten(x, start_dim=1)
        return self.linear_block(x)

    def __create_conv_layers(self, cfg):
        layers = []
        in_ch = self.cfg.in_channels

        for x in self.cfg.architecture:
            if type(x) == tuple:
                # * convolution
                layers += [
                    ConvLayer(
                        in_channels=in_ch, out_channels=x[1],
                        kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_ch = x[1]
            elif x == "maxpool":
                # * add max pooling layer
                layers += [nn.MaxPool2d(2,2)]
            elif type(x) == list:
                # * ConvBlock
                convs = x[:-1]
                num_repeat = x[-1]
                
                for _ in range(num_repeat):
                    for conv in convs:
                        layers += [
                            ConvLayer(
                                in_channels=in_ch, out_channels=conv[1],
                                kernel_size=conv[0], stride=conv[2], padding=conv[3]
                            )
                        ]
                        in_ch = conv[1]

        return nn.Sequential(*layers)

    def __create_linear_block(self, cfg):
        S, B, C = cfg.split_size, cfg.num_boxes, cfg.num_classes

        linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 512), # * og paper uses 4096
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S * S * (C + B * 5)) # * Reshape to S * S * (num_classes + num_boxes * (box_prob, coords))
        )

        return linear_block



# Tests

def test_YOLOv1(S=7, B=2, C=20):
    yolo_cfg = YoloConfig(3, S, B, C)
    model = YOLOv1(yolo_cfg)
    x= torch.randn(2,3,448,448)
    # print(model(x).shape)
    print(summary(model, x, batch_size=2, show_input=True))

# test_YOLOv1()