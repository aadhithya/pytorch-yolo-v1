import torch


def get_iou(box1, box2, box_format="midpoint"):
    """
        Calculates intersection over union
        Parameters:
            boxes1 (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            box2 (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
    """

    if box_format == "corners":
        # * box 1
        box1_x1 = box1[...,0:1]
        box1_y1 = box1[...,1:2]
        box1_x2 = box1[...,2:3]
        box1_y2 = box1[...,3:4]

        # * box 2
        box2_x1 = box2[...,0:1]
        box2_y1 = box2[...,1:2]
        box2_x2 = box2[...,2:3]
        box2_y2 = box2[...,3:4]

    elif box_format == "midpoint":
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
    else:
        raise NotImplementedError(f"OOPs! {box_format} not supported!")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1)) 

    retrun intersection / (box1_area + box2_area - intersection + 1e-6)


def nonmax_suppression(bboxes, iou_threshold, prob_threshold, box_format="midpoint"):
    # bboxes: [[class, pblty, x1, y1, x2, y2],[...],...,[...]]

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = boxes.pop(0)

        # * keep boxes that don't belong to the same class as chosen_box or 
        # * add the boxes if the iou with chosen box is  less than threshold
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box or
            get_iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms