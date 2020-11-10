import torch
from torch.utils.data import Dataset

import os
import pandas as pd
from PIL import Image

class VOCDataset(Dataset):
    def __init__(
        self, csv_dir, img_dir,
        label_dir, S=7, B=2, C=20,
        transforms=None
    ):
        super().__init__()
        self.annotations = pd.read_csv(csv_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = 20

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir,self.annotations.iloc[idx, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_lbl, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n","")
                ]

                boxes += [[class_lbl, x, y, w, h]]

        img_path = os.path.join(self.img_dir,self.annotations.iloc[idx, 0])

        img = Image.open(img_path)

        label_matrix = torch.zeros(self.S, self.S, self.C + 5*self.B)  # SxSx30

        for box in  boxes:
            lbl, x, y, w, h = box
            lbl = int(lbl)

            # find which cell the x, y is in
            i, j = int(self.S * x), int(self.S * y) 
            # find x and y pos relative to the cell
            x_cell, y_cell = self.S * x - i, self.S * y -j 
            # find height and width relative to cell.
            w_cell, h_cell = (
                w * self.S,
                h * self.S
            )

            if label_matrix[i, j, 20] == 0:
                # presence of object
                label_matrix[i, j, 20] = 1
                # position of obj
                label_matrix[i, j, 21:25] = torch.tensor(x_cell, y_cell, w_cell, h_cell)
                # class label
                label_matrix[i, j, lbl] = 1

        return img, label_matrix

