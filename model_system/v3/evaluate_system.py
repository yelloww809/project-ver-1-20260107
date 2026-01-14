import numpy as np
import matplotlib
# [核心修复] 强制使用非交互式后端，不需要GUI支持，速度更快且不会报错
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
OUTPUT_DIR = Path(r'E:\huangwenhao\test_results\test_results_v3_2') # 结果保存路径
# 请确保这两个路径指向你真实的 .pt 文件
LARGE_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_large_jpg_1\weights\best.pt'
SMALL_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_small_jpg_1_epochs40\weights\best.pt'
DEVICE = 'cuda:0'
FONT_SIZE = 8

def phys2pix_eval(val_t, val_f, duration_ms, f_range, img_w, img_h):
    """物理坐标 -> 绘图用像素坐标"""
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_thresholds):
    """
    计算单张图的 TP 矩阵，严格按类别隔离匹配。
    输入输出均使用物理坐标 (ms, MHz)，无需转回像素坐标。
    """
    # 1. 准备物理参数
    # 在这里我们直接使用物理坐标计算 IoU，不需要转换到像素
    
    # 2. 整理 Preds (物理坐标)
    # 格式: [t1, f1, t2, f2, conf, cls]
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

        # 计算物理坐标下的 IoU [N_pred, M_gt]
        iou = box_iou(cls_g_boxes[:, :4], cls_p_boxes[:, :4]) 

        # 对每个阈值进行匹配
        for i, iou_thr in enumerate(iou_thresholds):
            matches = torch.nonzero(iou >= iou_thr) 
            if matches.shape[0] > 0:
                match_vals = iou[matches[:, 0], matches[:, 1]]
                matches = torch.cat([matches, match_vals[:, None]], 1)
                
                # 按 IoU 降序
                matches = matches[matches[:, 2].argsort(descending=True)]
                # Pred 去重
                _, unique_p_idx = np.unique(matches[:, 1].cpu().numpy(), return_index=True)
                matches = matches[unique_p_idx]
                # GT 去重
                _, unique_g_idx = np.unique(matches[:, 0].cpu().numpy(), return_index=True)
                matches = matches[unique_g_idx]
                
                # 填回 correct
                relative_pred_indices = matches[:, 1].long()
                absolute_pred_indices = p_indices[relative_pred_indices]
                correct[absolute_pred_indices, i] = True

    return correct, p_boxes[:, 4], p_boxes[:, 5], g_boxes[:, 4]

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL, device=DEVICE)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    
    # # 如果文件太多，可以先测试 10 个
    # bin_files = bin_files[:10]
    
    stats = [] 
    # 定义评估用的 IoU 阈值: 0.5 - 0.95
    iou_v = torch.linspace(0.5, 0.95, 10)
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    
    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        sample_dir = OUTPUT_DIR / bin_file.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 运行预测
        preds, img_large, img_dims, meta, slices = system.predict(
            bin_file, json_file, 
            conf_thres=0.2,  
            iou_thres=0.7      
        )
        
        # 2. 保存中间结果 (可选)
        cv2.imwrite(str(sample_dir / f"{bin_file.stem}_large.jpg"), img_large)
        for img_slice, suffix in slices:
            cv2.imwrite(str(sample_dir / f"{bin_file.stem}{suffix}.jpg"), img_slice)
        system.save_results(preds, sample_dir / f"{bin_file.stem}_pred.json")
            
        # 3. 计算指标统计量
        gts = meta.get('signals', [])
        correct, conf, pcls, tcls = compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_v)
        # 将结果转回 CPU 以便后续拼接
        stats.append((correct.cpu(), conf.cpu(), pcls.cpu(), tcls.cpu()))
        
        # 4. 可视化 (仅画高分框，避免图片被淹没)
        # 准备绘图参数
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
        plt.axis('off'); plt.savefig(sample_dir / f"{bin_file.stem}_GT.jpg"); plt.close(fig_gt)
        
        # 画 Pred (红色/橙色) - 过滤 conf < 0.25
        fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
        ax_pred.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
        for p in preds:
            if p['confidence'] < 0.25: continue # <--- 仅为了可视化清晰
            x1, y1 = phys2pix_eval(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
            x2, y2 = phys2pix_eval(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
            color = 'red' if p['source'] == 'small' else 'orange'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(x1, y1-5, f"{p['class']} {p['confidence']:.2f}", color=color, fontsize=FONT_SIZE)
        plt.axis('off'); plt.savefig(sample_dir / f"{bin_file.stem}_Pred.jpg"); plt.close(fig_pred)

    # --- 最终指标计算 ---
    print("\n>>> Computing Final Metrics with Ultralytics...")
    # 拼接所有批次的数据 (此时它们还是 Tensor)
    stats = [torch.cat(x, 0) for x in zip(*stats)]
    tp, conf, pcls, tcls = stats
    
    if tp.shape[0] > 0:
        names = {i: str(i) for i in range(14)}
        
        # [关键修正] 调用前将 Tensor 转换为 Numpy
        results = ap_per_class(
            tp.numpy(), 
            conf.numpy(), 
            pcls.numpy(), 
            tcls.numpy(), 
            plot=False, 
            names=names
        )
        
        # 解析结果
        p, r, ap50, ap, unique_classes = results[2], results[3], results[5][:, 0], results[5].mean(1), results[6]
        
        print("\n" + "="*90)
        print(f"{'Class':<10} | {'P':<10} | {'R':<10} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}")
        print("-" * 90)
        # unique_classes = results[5]
        for i, c in enumerate(unique_classes):
            print(f"{int(c):<10} | {p[i]:.4f}     | {r[i]:.4f}     | {ap50[i]:.4f}     | {ap[i]:.4f}")
        print("-" * 90)
        print(f"{'ALL':<10} | {p.mean():.4f}     | {r.mean():.4f}     | {ap50.mean():.4f}     | {ap.mean():.4f}")
        print("=" * 90)
    else:
        print("No valid predictions found.")

if __name__ == "__main__":
    main()