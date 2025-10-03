import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points
from pointpillars.dataset import point_range_filter, data_augment


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
        # Kendi oluşturduğun info dosyası
        self.data_infos = read_pickle(os.path.join(data_root, f'radar_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())

        # Augmentation config (isteğe bağlı)
        self.data_aug_config = dict(
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            ),
            point_range_filter=[-3.0, 0.1, 0.0, 3.0, 7.0, 3.0],
            object_range_filter=[-3.0, 0.1, 0.0, 3.0, 7.0, 3.0]
        )

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]

        # Nokta bulutu dosya yolu
        radar_path = data_info['radar_path'].replace('velodyne', self.pts_prefix)
        pts_path = os.path.join(self.data_root, radar_path)
        pts = read_points(pts_path)  # Buradan (N,5): x,y,z,doppler,SNR

        # Anotasyonlar (senin JSON ya da pkl’deki ground truth’lar)
        annos_info = data_info['annos']
        annos_name = annos_info['name']
        annos_location = annos_info['location']  # (N,3)
        annos_dimension = annos_info['dimensions']  # (N,3)
        rotation_y = annos_info['rotation_y']  # (N,)

        # Ground truth box formatı: x,y,z,l,w,h,yaw
        gt_bboxes_3d = np.concatenate(
            [annos_location, annos_dimension, rotation_y[:, None]], axis=1
        ).astype(np.float32)

        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]

        data_dict = {
            'pts': pts.astype(np.float32),  # (N,5)
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels),
            'gt_names': annos_name
        }

        if self.split in ['train', 'trainval']:
            data_dict = data_augment(
                self.CLASSES, self.data_root, data_dict, self.data_aug_config
            )
        else:
            data_dict = point_range_filter(
                data_dict, point_range=self.data_aug_config['point_range_filter']
            )

        return data_dict

    def __len__(self):
        return len(self.data_infos)


if __name__ == '__main__':
    radar_data = RadarDataset(data_root='/mnt/ssd1/lifa_rdata/det/radar', 
                              split='train')
    sample = radar_data.__getitem__(0)
    print(sample['pts'].shape, sample['gt_bboxes_3d'].shape, sample['gt_labels'])
