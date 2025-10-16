import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

""" 
def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_img_list, batched_calib_list = [], []

    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)  # List[str]
        batched_img_list.append({})
        batched_calib_list.append({})

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list
    )

    return rt_data_dict
"""

def collate_fn(list_data):
    #print("\n--- [DEBUG] Batch İnceleniyor ---")
    culprit_found = False
    culprit_indices = []

    # Her batch'teki her bir örneği kontrol et
    for i, data_dict in enumerate(list_data):
        pts_shape = data_dict['pts'].shape
        # Az önce eklediğimiz 'debug_index' etiketini al
        original_index = data_dict.get('debug_index', 'Bilinmiyor')

        #print(f"  -> Batch'teki {i}. örnek (Orijinal Veri Seti İndeksi: {original_index}): Nokta Sayısı = {pts_shape[0]}")

        # Eğer nokta sayısı 0 ise, bu bizim aradığımız suçlu!
        if pts_shape[0] == 0:
            print(f"  !!!!!! SUÇLU BULUNDU !!!!!!")
            print(f"  !!!!!! Veri setindeki '{original_index}' numaralı örnek BOŞ !!!!!!")
            culprit_found = True
            culprit_indices.append(original_index)

    if culprit_found:
        print("--- İnceleme Bitti: Hatalı batch bulundu. Lütfen yukarıdaki indeks(ler)i kontrol edin. ---")
        # Programı burada durdurarak hatayı görmenizi sağlayabiliriz.
        # raise ValueError(f"Boş veri örnekleri bulundu: {culprit_indices}")

    # --- Bu kısım, sorunu kalıcı olarak çözen filtredir ---
    valid_list_data = [d for d in list_data if d['pts'] is not None and d['pts'].shape[0] > 0]
    if not valid_list_data:
        return None
    
    # Kodun geri kalanı sadece geçerli verilerle çalışacak
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_img_list, batched_calib_list = [], []

    for data_dict in valid_list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)
        batched_img_list.append({})
        batched_calib_list.append({})

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list
    )

    return rt_data_dict

def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=True):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader
