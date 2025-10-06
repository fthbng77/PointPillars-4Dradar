import os
import json
import numpy as np
import pickle
import random

# Ayarlar
BOX_SIZE = [0.5, 0.5, 1.0]     
CLASS_ID = 1                  
YAW_DEFAULT = 0.0           
SPLIT_RATIO = [0.8, 0.1, 0.1] 

# Klasör ayarları
input_folder = 'human3'       
output_folder = 'dataset_split_pkl' 
os.makedirs(output_folder, exist_ok=True)

# Tüm veriler burada toplanacak
all_frames = []

# Tüm JSON dosyaları içinde gez
for file_name in os.listdir(input_folder):
    if file_name.endswith('.json'):
        json_path = os.path.join(input_folder, file_name)

        with open(json_path, 'r') as f:
            try:
                raw = json.load(f)
            except Exception as e:
                print(f"⚠️ Hata: {file_name} okunamadı → {e}")
                continue

        frames = raw.get("data", [])
        print(f"📂 {file_name} → {len(frames)} frame bulundu")

        for i, frame in enumerate(frames):
            frame_data = frame.get("frameData", {})

            point_cloud = np.array(frame_data.get("pointCloud", []), dtype=np.float32)

            tracks = frame_data.get("trackData", [])
            gt_boxes_3d = []
            gt_labels_3d = []

            for t in tracks:
                x, y, z = t[1], t[2], t[3]
                dx, dy, dz = BOX_SIZE
                yaw = YAW_DEFAULT

                gt_boxes_3d.append([x, y, z, dx, dy, dz, yaw])
                gt_labels_3d.append(CLASS_ID)

            info = {
                "points": point_cloud,
                "gt_boxes_3d": np.array(gt_boxes_3d, dtype=np.float32),
                "gt_labels_3d": np.array(gt_labels_3d, dtype=np.int64),
                "frame_id": f"{file_name}_{i}"
            }

            all_frames.append(info)

print(f"📦 Toplam {len(all_frames)} frame toplandı.")

random.seed(42)
random.shuffle(all_frames)

n_total = len(all_frames)
n_train = int(SPLIT_RATIO[0] * n_total)
n_val   = int(SPLIT_RATIO[1] * n_total)
n_test  = n_total - n_train - n_val

train_data = all_frames[:n_train]
val_data   = all_frames[n_train:n_train+n_val]
test_data  = all_frames[n_train+n_val:]

with open(os.path.join(output_folder, 'radar_infos_train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(output_folder, 'radar_infos_val.pkl'), 'wb') as f:
    pickle.dump(val_data, f)
with open(os.path.join(output_folder, 'radar_infos_test.pkl'), 'wb') as f:
    pickle.dump(test_data, f)

print("✅ Train/Val/Test .pkl dosyaları başarıyla oluşturuldu:")
print(f" - Train: {len(train_data)} frame")
print(f" - Val  : {len(val_data)} frame")
print(f" - Test : {len(test_data)} frame")
