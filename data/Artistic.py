import copy
import logging
import cv2
import numpy
import torch
import numpy as np
import os
import Landmark2.utils as utils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Landmark2.Config import cfg


class StyleDataSet(Dataset):
    def __init__(self, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = os.path.join(root, 'images')
        self.number_landmarks = 68

        self.Fraction = cfg.W300.FRACTION

        self.Transform = transform
        self.root_landmark = os.path.join(root, 'landmarks')
        if is_train:
            self.annotation_file = os.path.join(root, 'train_list.txt')
        else:
            self.annotation_file = os.path.join(root, 'test_list_all.txt')

        self.database = self.get_image()

    def get_image(self):
        Data_base = []
        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()
        for temp_info in info_list:
            temp_name = os.path.join(self.root, temp_info)

            Points = np.load(os.path.join(self.root_landmark, temp_info[:-3] + "npy"))
            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'points': Points
                              })

        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['points']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        if self.Transform is not None:
            input = self.Transform(input)
        meta = {
            'image': input,
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'points': Points / (self.Image_size),
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'angle': 0.0,
            'Translation': [0.0, 0.0],
        }
        return meta


def get_cartoon_dataloader(batch_size, opt):
    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    train_dataset = StyleDataSet(root=opt.tgt_data, is_train=True, transform=normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    cartoon_dataset = StyleDataSet(root=opt.tgt_data, is_train=False, transform=normalize)
    cartoon_loader = torch.utils.data.DataLoader(
        cartoon_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, cartoon_loader
