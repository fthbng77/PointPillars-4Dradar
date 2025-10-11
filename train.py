import argparse
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pointpillars.utils import setup_seed
from pointpillars.dataset import RadarDataset, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def move_to_cuda(batch_dict):
    for key in batch_dict:
        for j, item in enumerate(batch_dict[key]):
            if torch.is_tensor(item):
                batch_dict[key][j] = item.cuda(non_blocking=True)
    return batch_dict


def main(args):
    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[INFO] Cihaz: {device}")

    # --- Dataset ve DataLoader ---
    train_dataset = RadarDataset(data_root=args.data_root, split='train')
    val_dataset = RadarDataset(data_root=args.data_root, split='val')
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = get_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False, drop_last=True)

    # --- Model ---
    pointpillars = PointPillars(nclasses=args.nclasses).to(device)
    loss_func = Loss()

    optimizer = torch.optim.AdamW(pointpillars.parameters(), lr=args.init_lr, betas=(0.95, 0.99), weight_decay=0.01)
    max_iters = len(train_loader) * args.max_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.init_lr * 10, total_steps=max_iters,
        pct_start=0.4, anneal_strategy='cos', cycle_momentum=True,
        base_momentum=0.95 * 0.895, max_momentum=0.95, div_factor=10
    )

    # --- Log & Checkpoint klasörleri ---
    os.makedirs(os.path.join(args.saved_path, 'summary'), exist_ok=True)
    os.makedirs(os.path.join(args.saved_path, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.saved_path, 'summary'))

    # --- Eğitim Döngüsü ---
    for epoch in range(args.max_epoch):
        print(f"\n========== EPOCH {epoch + 1}/{args.max_epoch} ==========")
        pointpillars.train()
        train_step = 0

        for batch_idx, data_dict in enumerate(tqdm(train_loader)):
            if device.type == "cuda":
                data_dict = move_to_cuda(data_dict)

            optimizer.zero_grad()
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            # --- Forward ---
            try:
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(
                    batched_pts=batched_pts, mode='train',
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels
                )
            except Exception as e:
                print(f"[WARN] Batch {batch_idx} atlandı (forward hatası): {e}")
                continue

            # --- Tensor boyutlarını düzenle ---
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)

            if pos_idx.sum() == 0:
                print(f"[WARN] Batch {batch_idx}: pozitif anchor yok, atlanıyor.")
                continue

            try:
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]
            except Exception as e:
                print(f"[WARN] Batch {batch_idx}: mask shape uyuşmazlığı ({e}), atlanıyor.")
                continue

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            # --- Kayıp hesapla ---
            loss_dict = loss_func(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_labels=batched_bbox_labels,
                num_cls_pos=num_cls_pos,
                batched_bbox_reg=batched_bbox_reg,
                batched_dir_labels=batched_dir_labels
            )

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Log yaz ---
            global_step = epoch * len(train_loader) + train_step + 1
            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'],
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1

        # --- Checkpoint ---
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            ckpt_path = os.path.join(args.saved_path, 'checkpoints', f'epoch_{epoch+1}.pth')
            torch.save(pointpillars.state_dict(), ckpt_path)
            print(f"[INFO] Checkpoint kaydedildi: {ckpt_path}")

        # --- Val ---
        if epoch % 2 != 0:
            pointpillars.eval()
            val_step = 0
            with torch.no_grad():
                for data_dict in tqdm(val_loader):
                    if device.type == "cuda":
                        data_dict = move_to_cuda(data_dict)
                    batched_pts = data_dict['batched_pts']
                    batched_gt_bboxes = data_dict['batched_gt_bboxes']
                    batched_labels = data_dict['batched_labels']

                    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(
                        batched_pts=batched_pts, mode='train',
                        batched_gt_bboxes=batched_gt_bboxes,
                        batched_gt_labels=batched_labels
                    )

                    bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                    bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                    batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                    batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                    batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                    batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

                    pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                    if pos_idx.sum() == 0:
                        continue

                    bbox_pred = bbox_pred[pos_idx]
                    batched_bbox_reg = batched_bbox_reg[pos_idx]
                    bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                    batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                    bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                    batched_dir_labels = batched_dir_labels[pos_idx]

                    num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                    bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                    batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                    batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                    loss_dict = loss_func(
                        bbox_cls_pred=bbox_cls_pred,
                        bbox_pred=bbox_pred,
                        bbox_dir_cls_pred=bbox_dir_cls_pred,
                        batched_labels=batched_bbox_labels,
                        num_cls_pos=num_cls_pos,
                        batched_bbox_reg=batched_bbox_reg,
                        batched_dir_labels=batched_dir_labels
                    )

                    global_step = epoch * len(val_loader) + val_step + 1
                    if global_step % args.log_freq == 0:
                        save_summary(writer, loss_dict, global_step, 'val')
                    val_step += 1

    print("[INFO] Eğitim tamamlandı.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radar PointPillars Eğitim')
    parser.add_argument('--data_root', default='/home/fatih/Xena Vision/PointPillars-4Dradar/pointpillars/dataset/pointpillar-ds',
                        help='RadarDataset dizini')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=2)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true', help='CUDA kapalı olsun mu')
    args = parser.parse_args()

    main(args)
