import numpy as np
import matplotlib
# [核心] 强制使用非交互式后端，防止大量绘图时 Tkinter 崩溃
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from inference_system import SignalInferenceSystem
# [核心] 导入官方计算函数
from ultralytics.utils.metrics import ap_per_class, box_iou

# ================= 配置 =================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test')  # Test集路径
OUTPUT_DIR = Path(r'E:\huangwenhao\test_results\test_results_v4')  # 结果保存路径

# 模型路径
LARGE_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_large_jpg_1\weights\best.pt'
SMALL_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_small_jpg_1_epochs40\weights\best.pt'
DEVICE = 'cuda:0'
FONT_SIZE = 8

# [新增] 可视化数量限制 (1-750)
# -1 表示全部可视化 (速度慢)
# 50 表示只可视化前 50 个 (推荐)
VIS_LIMIT = 10 

def phys2pix_eval(val_t, val_f, duration_ms, f_range, img_w, img_h):
    """物理坐标 -> 绘图用像素坐标"""
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_thresholds):
    """
    计算单张图的 TP 矩阵，严格按类别隔离匹配。
    """
    # 2. 整理 Preds (物理坐标)
    p_boxes = []
    for p in preds:
        p_boxes.append([p['start_time'], p['start_frequency'], p['end_time'], p['end_frequency'], p['confidence'], p['class']])
    p_boxes = torch.tensor(p_boxes) if len(p_boxes) > 0 else torch.zeros((0, 6))

    # 3. 整理 GTs (物理坐标)
    g_boxes = []
    for g in gts:
        g_boxes.append([g['start_time'], g['start_frequency'], g['end_time'], g['end_frequency'], g['class']])
    g_boxes = torch.tensor(g_boxes) if len(g_boxes) > 0 else torch.zeros((0, 5))

    # 4. 初始化
    correct = torch.zeros(len(p_boxes), len(iou_thresholds), dtype=torch.bool)
    
    if len(p_boxes) == 0:
        return correct, torch.tensor([]), torch.tensor([]), g_boxes[:, 4] if len(g_boxes) > 0 else torch.tensor([])

    # 5. 按类别匹配
    unique_classes = torch.unique(torch.cat([p_boxes[:, 5], g_boxes[:, 4]])) if len(g_boxes) > 0 else torch.unique(p_boxes[:, 5])

    for cls in unique_classes:
        # 筛选同类框
        p_mask = (p_boxes[:, 5] == cls)
        g_mask = (g_boxes[:, 4] == cls)
        
        cls_p_boxes = p_boxes[p_mask]
        cls_g_boxes = g_boxes[g_mask]
        
        p_indices = torch.where(p_mask)[0] # 记录原始索引

        if len(cls_g_boxes) == 0 or len(cls_p_boxes) == 0:
            continue

        # 计算物理坐标下的 IoU
        iou = box_iou(cls_g_boxes[:, :4], cls_p_boxes[:, :4]) 

        # 对每个阈值进行匹配
        for i, iou_thr in enumerate(iou_thresholds):
            matches = torch.nonzero(iou >= iou_thr) 
            if matches.shape[0] > 0:
                match_vals = iou[matches[:, 0], matches[:, 1]]
                matches = torch.cat([matches, match_vals[:, None]], 1)
                
                # 双重排序去重 (Ultralytics 标准逻辑)
                matches = matches[matches[:, 2].argsort(descending=True)]
                _, unique_p_idx = np.unique(matches[:, 1].cpu().numpy(), return_index=True)
                matches = matches[unique_p_idx]
                matches = matches[matches[:, 2].argsort(descending=True)]
                _, unique_g_idx = np.unique(matches[:, 0].cpu().numpy(), return_index=True)
                matches = matches[unique_g_idx]
                
                relative_pred_indices = matches[:, 1].long()
                absolute_pred_indices = p_indices[relative_pred_indices]
                correct[absolute_pred_indices, i] = True

    return correct, p_boxes[:, 4], p_boxes[:, 5], g_boxes[:, 4]

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL, device=DEVICE)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    stats = [] 
    iou_v = torch.linspace(0.5, 0.95, 10)
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    print(f">>> Visualization Limit: {VIS_LIMIT}")
    
    vis_count = 0 # 计数器

    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        # 判断是否需要可视化
        is_visualizing = (VIS_LIMIT == -1) or (vis_count < VIS_LIMIT)

        # 1. 运行预测 (必须做)
        # conf=0.001 保证指标计算准确
        preds, img_large, img_dims, meta, slices = system.predict(
            bin_file, json_file, 
            conf_thres=0.001,  
            iou_thres=0.6      
        )
        
        # 2. 保存逻辑 (分流)
        if is_visualizing:
            # 方式 A: 创建专属文件夹，保存所有图片和JSON
            sample_dir = OUTPUT_DIR / bin_file.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存原图和切片
            cv2.imwrite(str(sample_dir / f"{bin_file.stem}_large.jpg"), img_large)
            for img_slice, suffix in slices:
                cv2.imwrite(str(sample_dir / f"{bin_file.stem}{suffix}.jpg"), img_slice)
            
            # 保存 JSON
            system.save_results(preds, sample_dir / f"{bin_file.stem}_pred.json")
        else:
            # 方式 B: 不创建文件夹，直接保存 JSON 到根目录 (节省时间和空间)
            system.save_results(preds, OUTPUT_DIR / f"{bin_file.stem}_pred.json")

        # 3. 计算指标统计量 (必须做)
        gts = meta.get('signals', [])
        correct, conf, pcls, tcls = compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_v)
        stats.append((correct.cpu(), conf.cpu(), pcls.cpu(), tcls.cpu()))
        
        # 4. 可视化绘图 (仅在计数额度内执行)
        if is_visualizing:
            h, w = img_dims[1], img_dims[0]
            obs = meta['observation_range']
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            iq_signal = raw_data[::2] + 1j * raw_data[1::2]
            fs = (obs[1] - obs[0]) * 1e6
            duration_ms = (len(iq_signal) / fs) * 1000
            
            # 画 GT (绿色)
            fig_gt, ax_gt = plt.subplots(figsize=(12, 12 * h / w))
            ax_gt.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
            for g in gts:
                x1, y1 = phys2pix_eval(g['start_time'], g['start_frequency'], duration_ms, obs, w, h)
                x2, y2 = phys2pix_eval(g['end_time'], g['end_frequency'], duration_ms, obs, w, h)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
                ax_gt.add_patch(rect)
                ax_gt.text(x1, y1-5, f"{g['class']}", color='lime', fontsize=FONT_SIZE)
            plt.axis('off')
            plt.savefig(sample_dir / f"{bin_file.stem}_GT.jpg")
            plt.close(fig_gt)
            
            # 画 Pred (红色/橙色) - 过滤低分框以免看不清
            fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
            ax_pred.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
            for p in preds:
                if p['confidence'] < 0.25: continue 
                x1, y1 = phys2pix_eval(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
                x2, y2 = phys2pix_eval(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
                color = 'red' if p['source'] == 'small' else 'orange'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax_pred.add_patch(rect)
                ax_pred.text(x1, y1-5, f"{p['class']} {p['confidence']:.2f}", color=color, fontsize=FONT_SIZE)
            plt.axis('off')
            plt.savefig(sample_dir / f"{bin_file.stem}_Pred.jpg")
            plt.close(fig_pred)

            vis_count += 1 # 增加计数

    # --- 最终指标计算 ---
    print("\n>>> Computing Final Metrics with Ultralytics...")
    stats = [torch.cat(x, 0) for x in zip(*stats)]
    tp, conf, pcls, tcls = stats
    
    if tp.shape[0] > 0:
        names = {i: str(i) for i in range(14)}
        
        # 将 Tensor 转为 Numpy
        results = ap_per_class(
            tp.numpy(), 
            conf.numpy(), 
            pcls.numpy(), 
            tcls.numpy(), 
            plot=False, 
            names=names
        )
        
        # [修正索引与维度]
        # Index 2: P, Index 3: R, Index 5: AP Matrix, Index 6: Classes
        p, r = results[2], results[3]
        ap_matrix = results[5]
        unique_classes = results[6]

        # 维度安全检查 (处理单类别 1D 数组情况)
        if ap_matrix.ndim == 1:
            ap_matrix = ap_matrix[None, :]

        ap50 = ap_matrix[:, 0]
        ap_mean = ap_matrix.mean(1)
        
        print("\n" + "="*90)
        print(f"{'Class':<10} | {'P':<10} | {'R':<10} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}")
        print("-" * 90)
        
        for i, c in enumerate(unique_classes):
            print(f"{int(c):<10} | {p[i]:.4f}     | {r[i]:.4f}     | {ap50[i]:.4f}     | {ap_mean[i]:.4f}")
        print("-" * 90)
        print(f"{'ALL':<10} | {p.mean():.4f}     | {r.mean():.4f}     | {ap50.mean():.4f}     | {ap_mean.mean():.4f}")
        print("=" * 90)
    else:
        print("No valid predictions found.")

if __name__ == "__main__":
    main()