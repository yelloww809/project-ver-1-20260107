import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import shutil
import torch
import seaborn as sns # 用于画混淆矩阵热力图
from pathlib import Path
from tqdm import tqdm
from inference_system import SignalInferenceSystem, SCHEME
from ultralytics.utils.metrics import ap_per_class, box_iou

# ================= 配置区域 =================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test')

# 自动根据 SCHEME 生成输出文件夹名称
# SCHEME_NAME = "base" if SCHEME == 0 else "dual_large_small"
# OUTPUT_DIR = Path(rf'E:\huangwenhao\test_results\test_results_v7_{SCHEME_NAME}')
OUTPUT_DIR = Path(rf'E:\huangwenhao\results\v9\system_v7\test_v7_sh{SCHEME}_1')

# 模型路径
BASE_MODEL = r'E:\huangwenhao\results\v9\train\train_v9_base_2_1\weights\best.pt' # Scheme 0 用
LARGE_MODEL = r'E:\huangwenhao\results\v9\train\train_v9_large_2_1\weights\best.pt' # Scheme 1 用
SMALL_MODEL = r'E:\huangwenhao\results\v9\train\train_v9_small_1_1\weights\best.pt' # Scheme 1 用

DEVICE = 'cuda:0'
FONT_SIZE = 8

# [阈值设置]
METRIC_CONF_THRES = 0.20  # 计算 mAP 用
VIS_CONF_THRES = 0.20      # 可视化 & 混淆矩阵用
VIS_LIMIT = 50 

# ================= 混淆矩阵类 =================
class ConfusionMatrix:
    def __init__(self, num_classes, conf_thres=0.20, iou_thres=0.5):
        self.nc = num_classes
        self.conf = conf_thres
        self.iou = iou_thres
        # 矩阵大小: (nc + 1) x (nc + 1), 最后一列/行为 Background
        self.matrix = np.zeros((self.nc + 1, self.nc + 1))
        
        # 坐标缩放因子 (复用匹配逻辑中的 Scaling，确保 IoU 一致)
        self.SCALE_TIME = 1000.0
        self.SCALE_FREQ = 1.0

    def process_batch(self, preds, gts):
        """
        更新混淆矩阵。
        注意：这里只处理 conf > self.conf 的框。
        """
        # 1. 过滤 & 缩放 Preds
        p_clean = [p for p in preds if p['confidence'] >= self.conf]
        
        # 转换为 Scaled Tensor
        def to_scaled(item_list, is_gt=False):
            boxes = []
            for item in item_list:
                t1, t2 = min(item['start_time'], item['end_time']), max(item['start_time'], item['end_time'])
                f1, f2 = min(item['start_frequency'], item['end_frequency']), max(item['start_frequency'], item['end_frequency'])
                x1, x2 = t1 * self.SCALE_TIME, t2 * self.SCALE_TIME
                y1, y2 = f1 * self.SCALE_FREQ, f2 * self.SCALE_FREQ
                
                if is_gt: boxes.append([item['class'], x1, y1, x2, y2])
                else: boxes.append([x1, y1, x2, y2, item['confidence'], item['class']])
            
            t = torch.tensor(boxes, dtype=torch.float32)
            # Pred 按 conf 降序 (虽然 CM 不严格依赖顺序，但保持一致好)
            if not is_gt and len(boxes) > 0:
                t = t[t[:, 4].argsort(descending=True)]
            return t

        pred_t = to_scaled(p_clean, is_gt=False)
        gt_t = to_scaled(gts, is_gt=True)

        # 2. 匹配逻辑 (Greedy by IoU, 简化版用于 CM)
        # 混淆矩阵通常允许 Cross-class matching 来统计误判
        
        if len(pred_t) == 0:
            # 所有 GT 都是 FN (Background)
            if len(gt_t) > 0:
                for cls_g in gt_t[:, 0]:
                    self.matrix[int(cls_g), self.nc] += 1
            return

        if len(gt_t) == 0:
            # 所有 Pred 都是 FP (Background)
            for cls_p in pred_t[:, 5]:
                self.matrix[self.nc, int(cls_p)] += 1
            return

        # 计算 IoU [N_pred, N_gt]
        iou = box_iou(pred_t[:, :4], gt_t[:, 1:])
        
        # 匹配
        matches = torch.nonzero(iou >= self.iou)
        
        if len(matches) > 0:
            match_iou = iou[matches[:, 0], matches[:, 1]]
            matches = torch.cat([matches, match_iou[:, None]], 1)
            # 按 IoU 降序排序
            matches = matches[matches[:, 2].argsort(descending=True)]
            # 去重
            matches = matches[np.unique(matches[:, 1].numpy(), return_index=True)[1]] # unique GT
            matches = matches[matches[:, 2].argsort(descending=True)]
            matches = matches[np.unique(matches[:, 0].numpy(), return_index=True)[1]] # unique Pred
        else:
            matches = torch.zeros((0, 3))

        # 3. 填充矩阵
        matched_p_indices = matches[:, 0].long()
        matched_g_indices = matches[:, 1].long()
        
        # TP & 误判 (GT -> Pred)
        for idx in range(len(matches)):
            p_idx, g_idx = int(matches[idx, 0]), int(matches[idx, 1])
            gt_cls = int(gt_t[g_idx, 0])
            pred_cls = int(pred_t[p_idx, 5])
            self.matrix[gt_cls, pred_cls] += 1
            
        # FN (未匹配的 GT -> Background)
        all_g_indices = set(range(len(gt_t)))
        matched_g_set = set(matched_g_indices.numpy())
        for g_idx in (all_g_indices - matched_g_set):
            gt_cls = int(gt_t[g_idx, 0])
            self.matrix[gt_cls, self.nc] += 1
            
        # FP (未匹配的 Pred -> Background)
        all_p_indices = set(range(len(pred_t)))
        matched_p_set = set(matched_p_indices.numpy())
        for p_idx in (all_p_indices - matched_p_set):
            pred_cls = int(pred_t[p_idx, 5])
            self.matrix[self.nc, pred_cls] += 1

    def plot(self, save_dir, names):
        # 1. 原始计数矩阵
        self._plot_matrix(self.matrix, save_dir / "confusion_matrix_raw.jpg", names, "Confusion Matrix (Raw)", fmt='.0f')
        
        # 2. 归一化矩阵 (Row-normalized)
        # 加上 1e-6 防止除以 0
        matrix_norm = self.matrix / (self.matrix.sum(1)[:, None] + 1e-6)
        self._plot_matrix(matrix_norm, save_dir / "confusion_matrix_norm.jpg", names, "Confusion Matrix (Normalized)", fmt='.2f')

    def _plot_matrix(self, matrix, save_path, names, title, fmt):
        plt.figure(figsize=(12, 10))
        # 轴标签包括 "Background"
        labels = list(names.values()) + ['Background']
        
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, square=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

# ================= 核心 mAP 计算 (v6_2 Strict Logic) =================
def process_batch_metrics_strict(preds, gts, meta, iou_thresholds):
    """
    使用数值缩放 + 置信度优先匹配。
    """
    SCALE_TIME = 1000.0 
    SCALE_FREQ = 1.0
    
    def to_scaled_tensor(item_list, is_gt=False):
        boxes = []
        for item in item_list:
            t1, t2 = min(item['start_time'], item['end_time']), max(item['start_time'], item['end_time'])
            f1, f2 = min(item['start_frequency'], item['end_frequency']), max(item['start_frequency'], item['end_frequency'])
            x1, x2 = t1 * SCALE_TIME, t2 * SCALE_TIME
            y1, y2 = f1 * SCALE_FREQ, f2 * SCALE_FREQ
            
            if is_gt: boxes.append([item['class'], x1, y1, x2, y2])
            else: boxes.append([x1, y1, x2, y2, item['confidence'], item['class']])
        
        t = torch.tensor(boxes, dtype=torch.float32)
        if not is_gt and len(boxes) > 0:
            # 严格 Sort by Conf
            t = t[t[:, 4].argsort(descending=True)]
        return t

    pred_t = to_scaled_tensor(preds, is_gt=False)
    gt_t = to_scaled_tensor(gts, is_gt=True)
    
    tp = torch.zeros(pred_t.shape[0], iou_thresholds.shape[0], dtype=torch.bool)
    
    if pred_t.shape[0] == 0:
        return tp, torch.tensor([]), torch.tensor([]), gt_t[:, 0] if len(gts) > 0 else torch.tensor([])
    if gt_t.shape[0] == 0:
        return tp, pred_t[:, 4], pred_t[:, 5], torch.tensor([])

    iou_matrix = box_iou(pred_t[:, :4], gt_t[:, 1:])

    for i, iou_thresh in enumerate(iou_thresholds):
        matches = torch.nonzero(iou_matrix >= iou_thresh)
        if matches.shape[0] > 0:
            match_cls = pred_t[matches[:, 0], 5] == gt_t[matches[:, 1], 0]
            matches = matches[match_cls]
            if matches.shape[0] > 0:
                match_iou = iou_matrix[matches[:, 0], matches[:, 1]]
                matches = torch.cat([matches, match_iou[:, None]], 1)
                
                # Strict Priority: 
                # 1. Best IoU for each Pred
                matches = matches[matches[:, 2].argsort(descending=True)]
                _, unique_pred_idx = np.unique(matches[:, 0].numpy(), return_index=True)
                unique_pred_idx = np.sort(unique_pred_idx) # Restore Conf Order
                matches = matches[unique_pred_idx]
                
                # 2. Conf Priority for each GT
                # Since matches is sorted by Pred Conf, first arrival takes GT
                _, unique_gt_idx = np.unique(matches[:, 1].numpy(), return_index=True)
                unique_gt_idx = np.sort(unique_gt_idx)
                matches = matches[unique_gt_idx]
                
                tp[matches[:, 0].long(), i] = True

    return tp, pred_t[:, 4], pred_t[:, 5], gt_t[:, 0]

# ================= 可视化逻辑 =================
def phys2pix_vis(val_t, val_f, duration_ms, f_range, img_w, img_h):
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def main():
    if SCHEME == 0:
        system = SignalInferenceSystem(base_model_path=BASE_MODEL, device=DEVICE)
    else:
        system = SignalInferenceSystem(large_model_path=LARGE_MODEL, small_model_path=SMALL_MODEL, device=DEVICE)
        
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    stats = [] 
    iou_v = torch.linspace(0.5, 0.95, 10)
    
    # 初始化混淆矩阵 (14类)
    cm = ConfusionMatrix(num_classes=14, conf_thres=VIS_CONF_THRES, iou_thres=0.5)
    
    print(f">>> Start Testing V7 (SCHEME={SCHEME}) on {len(bin_files)} files...")
    
    vis_count = 0 
    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        sample_dir = OUTPUT_DIR / bin_file.stem
        is_visualizing = (VIS_LIMIT == -1) or (vis_count < VIS_LIMIT)

        # 1. 预测 (METRIC_CONF_THRES = 0.001)
        preds, img_vis, img_dims, meta, slices = system.predict(
            bin_file, json_file, conf_thres=METRIC_CONF_THRES, iou_thres=0.6
        )
        
        # 2. 计算指标 & 更新 CM
        gts = meta.get('signals', [])
        
        # mAP stats
        tp, conf, pcls, tcls = process_batch_metrics_strict(preds, gts, meta, iou_v)
        stats.append((tp.cpu(), conf.cpu(), pcls.cpu(), tcls.cpu()))
        
        # Confusion Matrix (使用 VIS_CONF_THRES 过滤后的 preds)
        cm.process_batch(preds, gts)
        
        # 3. 可视化
        if is_visualizing:
            sample_dir.mkdir(parents=True, exist_ok=True)
            vis_preds = [p for p in preds if p['confidence'] >= VIS_CONF_THRES]
            
            cv2.imwrite(str(sample_dir / f"{bin_file.stem}_main.jpg"), img_vis)
            for img_slice, suffix in slices:
                cv2.imwrite(str(sample_dir / f"{bin_file.stem}{suffix}.jpg"), img_slice)
            system.save_results(vis_preds, sample_dir / f"{bin_file.stem}_pred.json")
            
            h, w = img_dims[1], img_dims[0]
            obs = meta['observation_range']
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            iq_signal = raw_data[::2] + 1j * raw_data[1::2]
            fs = (obs[1] - obs[0]) * 1e6
            duration_ms = (len(iq_signal) / fs) * 1000
            
            # GT Plot
            fig_gt, ax_gt = plt.subplots(figsize=(12, 12 * h / w))
            ax_gt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            for g in gts:
                x1, y1 = phys2pix_vis(g['start_time'], g['start_frequency'], duration_ms, obs, w, h)
                x2, y2 = phys2pix_vis(g['end_time'], g['end_frequency'], duration_ms, obs, w, h)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
                ax_gt.add_patch(rect)
                ax_gt.text(x1, y1-5, f"{g['class']}", color='lime', fontsize=FONT_SIZE)
            plt.axis('off')
            plt.savefig(sample_dir / f"{bin_file.stem}_GT.jpg")
            plt.close(fig_gt)
            
            # Pred Plot
            fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
            ax_pred.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            for p in vis_preds:
                x1, y1 = phys2pix_vis(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
                x2, y2 = phys2pix_vis(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
                color = 'red' if p['source'] == 'small' else 'orange'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax_pred.add_patch(rect)
                ax_pred.text(x1, y1-5, f"{p['class']} {p['confidence']:.2f}", color=color, fontsize=FONT_SIZE)
            plt.axis('off')
            plt.savefig(sample_dir / f"{bin_file.stem}_Pred.jpg")
            plt.close(fig_pred)
            vis_count += 1 

    print("\n>>> Computing Final Metrics...")
    
    # 保存混淆矩阵
    names = {i: str(i) for i in range(14)}
    cm.plot(OUTPUT_DIR, names)
    print(f"[Success] Confusion Matrix saved to {OUTPUT_DIR}")

    if not stats: return
    stats = [torch.cat(x, 0) for x in zip(*stats)]
    tp, conf, pcls, tcls = stats
    if tp.shape[0] > 0:
        results = ap_per_class(tp.numpy(), conf.numpy(), pcls.numpy(), tcls.numpy(), plot=False, names=names)
        p, r, ap50, ap_mean = results[2], results[3], results[5][:, 0], results[5].mean(1)
        
        result_txt_path = OUTPUT_DIR.with_suffix('.txt')
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"SCHEME: {SCHEME} | METRIC_CONF: {METRIC_CONF_THRES}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Class':<10} | {'P':<10} | {'R':<10} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}\n")
            for i, c in enumerate(results[6]):
                f.write(f"{int(c):<10} | {p[i]:.4f}     | {r[i]:.4f}     | {ap50[i]:.4f}     | {ap_mean[i]:.4f}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'ALL':<10} | {p.mean():.4f}     | {r.mean():.4f}     | {ap50.mean():.4f}     | {ap_mean.mean():.4f}\n")
        print(f"[Success] Metrics saved to: {result_txt_path}")

if __name__ == "__main__":
    main()