import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points
# augment ve filtreyi import etmeye gerek yok, kaldırıyoruz


class RadarDataset(Dataset):
    CLASSES = {
        'drone': 0,
        'person': 1,
    }

    def __init__(self, data_root, split, pts_prefix='radar_points'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix

        self.data_infos = read_pickle(os.path.join(data_root, f'radar_infos_{split}.pkl'))

        if isinstance(self.data_infos, dict):
            self.sorted_ids = list(self.data_infos.keys())
        else:
            self.sorted_ids = list(range(len(self.data_infos)))  # index listesi

    def __getitem__(self, index):
        # data_info çek
        if isinstance(self.data_infos, dict):
            data_info = self.data_infos[self.sorted_ids[index]]
        else:
            data_info = self.data_infos[index]

        pts = data_info['points'] 
        gt_bboxes_3d = data_info['gt_boxes_3d']
        gt_labels_3d = data_info['gt_labels_3d']

        # gt_names: label ID’ye göre isim üret
        gt_names = []
        for lbl in gt_labels_3d:
            for k, v in self.CLASSES.items():
                if v == lbl:
                    gt_names.append(k)
                    break

        data_dict = {
            'pts': pts.astype(np.float32),
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels_3d,
            'gt_names': gt_names
        }

        # augment veya filtre yok
        return data_dict

    def __len__(self):
        return len(self.sorted_ids)


if __name__ == '__main__':
    radar_data = RadarDataset(data_root='/home/fatih/pointpillar-ds', 
                              split='train')
    sample = radar_data.__getitem__(0)
    print(sample['pts'].shape, sample['gt_bboxes_3d'].shape, sample['gt_labels'])
