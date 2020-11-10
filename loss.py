from model import YOLOv1
import torch
import torch.nn as nn

class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        """
        __init__ initialize YOLOv1 Loss.

        Args:
            S (int, optional): split_size. Defaults to 7.
            B (int, optional): number of boxes. Defaults to 2.
            C (int, optional): number of classes. Defaults to 20.
        """ 
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.l_noobl = 0.5
        self.l_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + Self.B*5)

        iou_b1 = get_iou(predictions[...,21:25], target[...,21:25])
        iou_b2 = get_iou(predictions[...,26:30], target[...,21:25])
        ious = torch.stack([iou_b1, iou_b2], 0)

        _, max_iou = torch.max(ious, dim=0)
        exists_box = target[...,20].unsqueeze(3) # select target objectness.object

        # * Box Coordinates Loss

        # Select the bounding boxes with highest IoU
        box_predictions = exists_box * (
            (
                max_iou * predictions[..., 26:30] +
                (1 - max_iou) * predictions[..., 21:25]
            )
        )

        # Select targets which has an object
        box_targets = exists_box * target[...,21:25]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # * Object Losss

        pred_box = (
            max_iou * predictions[..., 25:26] + 
            (1-max_iou) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        # * No Object Loss
        # For the first box
        no_boject_loss = self.mse(
            torch.flatten((1-max_iou) * predictions[...,20:21], start_dim=1),
            torch.flatten((1-max_iou) * target[...,20:21], start_dim=1)
        )
        # For the second box
        no_boject_loss += self.mse(
            torch.flatten(max_iou * predictions[...,25:26], start_dim=1),
            torch.flatten(max_iou * target[...,20:21], start_dim=1)
        )
        # * Class prediction Loss

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[...,:20], end_dim=-2),
            torch.flatten(exists_box * target[...,:20], end_dim=-2)
        )

        # * Total Loss
        loss = (
            self.l_coord * box_loss 
            + object_loss
            + self.l_noobl * no_boject_loss
            + class_loss
        )

        return loss
