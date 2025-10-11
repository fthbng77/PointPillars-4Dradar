import os
import pickle
import numpy as np

# ---- PKL dosyanın yolu ----
pkl_path = "/home/fatih/Xena Vision/PointPillars-4Dradar/pointpillars/dataset/pointpillar-ds/radar_infos_train.pkl"

# ---- Veriyi yükle ----
with open(pkl_path, "rb") as f:
    data_infos = pickle.load(f)

print(f"[INFO] Toplam örnek sayısı: {len(data_infos)}")

missing_files = []
empty_points = []
missing_labels = []
shape_mismatch = []
count_boxes_label_diff = 0

for i, info in enumerate(data_infos):
    # 1️⃣ Points verisini al
    if "points" in info:
        points_data = info["points"]
    elif "point_path" in info:
        points_data = info["point_path"]
    else:
        print(f"[WARN] {i}: 'points' anahtarı bulunamadı")
        continue

    # Eğer string (dosya yolu) ise
    if isinstance(points_data, str):
        if not os.path.exists(points_data):
            missing_files.append(i)
            continue
        try:
            pts = np.load(points_data)
        except Exception as e:
            print(f"[ERROR] {i}: {points_data} okunamadı ({e})")
            continue
    else:
        # points_data zaten array
        pts = np.array(points_data)

    if pts.ndim < 2 or pts.shape[0] == 0:
        empty_points.append(i)
        continue

    # 2️⃣ Ground truth
    gt_boxes = np.array(info.get("gt_boxes_3d", []))
    gt_labels = np.array(info.get("gt_labels_3d", []))

    if len(gt_boxes) == 0:
        missing_labels.append(i)
        continue

    if gt_boxes.shape[-1] != 7:
        shape_mismatch.append(i)
        continue

    if len(gt_boxes) != len(gt_labels):
        print(f"[WARN] {i}: bbox ({len(gt_boxes)}) != labels ({len(gt_labels)})")
        count_boxes_label_diff += 1

# ---- Sonuç Raporu ----
print("\n========== SONUÇ ==========")
print(f"Bozuk dosya yolu: {len(missing_files)} örnek")
print(f"Boş point cloud: {len(empty_points)} örnek")
print(f"GT kutusu/etiketi eksik: {len(missing_labels)} örnek")
print(f"Boyut uyuşmazlığı (gt_boxes.shape[-1] != 7): {len(shape_mismatch)} örnek")
print(f"Label ve box sayısı farklı: {count_boxes_label_diff} örnek")

if empty_points:
    print("\nBoş point cloud örnekleri (ilk 10):", empty_points[:10])
if missing_labels:
    print("\nEksik etiket örnekleri (ilk 10):", missing_labels[:10])
