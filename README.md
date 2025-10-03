# [RadarPillars: Efficient Object Detection from 4D Radar Point Clouds](https://arxiv.org/abs/2408.05020) 

A Simple PointPillars PyTorch Implenmentation for 4D Radar(TI) Detection. 


## Detection Visualization


## [Install] 

Install PointPillars as a python package and all its dependencies as follows:

```
cd PointPillars-4Dradar/
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install .
```

## [Training]

```
cd PointPillars-4Dradar/
python train.py --data_root your_path_to_dataset
```

## [Evaluation]

```
cd PointPillars-4Dradar/
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 
```

## [Test]

```
cd PointPillars-4Dradar/

# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 

# 2. infer and visualize point cloud detection and gound truth.
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path  --gt_path your_gt_path

# 3. infer and visualize point cloud & image detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --img_path your_img_path


e.g. 
a. [infer on val set 000134]

python test.py --ckpt pretrained/epoch_160.pth --pc_path pointpillars/dataset/demo_data/val/000134.bin

or

python test.py --ckpt pretrained/epoch_160.pth --pc_path pointpillars/dataset/demo_data/val/000134.bin \
               --calib_path pointpillars/dataset/demo_data/val/000134.txt \
               --img_path pointpillars/dataset/demo_data/val/000134.png \
               --gt_path pointpillars/dataset/demo_data/val/000134_gt.txt

b. [infer on test set 000002]

python test.py --ckpt pretrained/epoch_160.pth --pc_path pointpillars/dataset/demo_data/test/000002.bin

or 

python test.py --ckpt pretrained/epoch_160.pth --pc_path pointpillars/dataset/demo_data/test/000002.bin \
               --calib_path pointpillars/dataset/demo_data/test/000002.txt \
               --img_path pointpillars/dataset/demo_data/test/000002.png
```

## Acknowledements

Thanks for the open source code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).
