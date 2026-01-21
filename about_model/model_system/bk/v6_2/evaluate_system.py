import numpy as np
import matplotlib
# [核心] 强制使用非交互式后端
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

# ==============================================================================
#                                 CONFIGURATION
# ==============================================================================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test')  # Test集路径
OUTPUT_DIR = Path(r'E:\huangwenhao\results\v9\system\test_system_v6_2_large_2_1_small_1_1')  # 结果保存路径

# 模型路径
LARGE_MODEL = r'E:\huangwenhao\results\v9\train\train_v9_large_2_1\weights\best.pt'
SMALL_MODEL = r'E:\huangwenhao\results\v9\train\train_v9_small_1_1\weights\best.pt'
DEVICE = 'cuda:0'
FONT_SIZE = 8

# [评估参数] 
# 注意：这里设置的值会覆盖 inference_system.py 中的默认参数
CONF_THRES = 0.2
IOU_THRES = 0.70   

# [可视化限制]
VIS_LIMIT = 50 

# ==============================================================================
#                                 EVALUATION LOGIC
# ==============================================================================

def phys2pix_eval(val_t, val_f, duration_ms, f_range, img_w, img_h):
    """物理坐标 -> 绘图用像素坐标"""
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_thresholds):
    """
    [v6_2 修改版] 
    严格依据用户提供的《评分规则》步骤 3.1 进行匹配。
    核心变化：从 "IoU优先" 改为 "置信度优先"。
    """
    # --- 1. 数据准备 ---
    # [新增] 严格按照资料要求：先对预测框按 confidence 从高到低排序
    # 注意：这里的排序会改变 preds 的顺序，后续返回的 p_boxes 必须与这个顺序一致
    preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)

    # 整理 Preds (物理坐标) - 使用排序后的列表
    p_boxes = []
    for p in preds_sorted:
        p_boxes.append([p['start_time'], p['start_frequency'], p['end_time'], p['end_frequency'], p['confidence'], p['class']])
    p_boxes = torch.tensor(p_boxes) if len(p_boxes) > 0 else torch.zeros((0, 6))

    # 整理 GTs (物理坐标)
    g_boxes = []
    for g in gts:
        g_boxes.append([g['start_time'], g['start_frequency'], g['end_time'], g['end_frequency'], g['class']])
    g_boxes = torch.tensor(g_boxes) if len(g_boxes) > 0 else torch.zeros((0, 5))

    # 初始化 correct 矩阵 [N_pred, N_iou_thresh]
    correct = torch.zeros(len(p_boxes), len(iou_thresholds), dtype=torch.bool)
    
    if len(p_boxes) == 0:
        return correct, torch.tensor([]), torch.tensor([]), g_boxes[:, 4] if len(g_boxes) > 0 else torch.tensor([])
    if len(g_boxes) == 0:
        return correct, p_boxes[:, 4], p_boxes[:, 5], torch.tensor([])

    # --- 2. 执行匹配逻辑 (严格对应资料步骤 3.1) ---
    
    # 依然按类别隔离处理 (资料中提到: "找到所有与 Pi 类别相同的 GT")
    unique_classes = torch.unique(torch.cat([p_boxes[:, 5], g_boxes[:, 4]]))

    for cls in unique_classes:
        # 获取当前类别的索引 (mask)
        # 注意：p_indices 对应的是 p_boxes (已按 conf 降序) 中的索引
        p_mask = (p_boxes[:, 5] == cls)
        g_mask = (g_boxes[:, 4] == cls)
        
        # 筛选出当前类别的框
        cls_p_boxes = p_boxes[p_mask]
        cls_g_boxes = g_boxes[g_mask]
        
        # 记录它们在原始大列表中的索引，以便最后填入 correct 矩阵
        p_indices_in_global = torch.where(p_mask)[0]
        
        if len(cls_g_boxes) == 0 or len(cls_p_boxes) == 0:
            continue

        # 计算物理坐标下的 IoU [N_pred_cls, N_gt_cls]
        # 注意：这里我们依然保留 v6 的物理坐标计算，不做归一化，仅修改匹配逻辑
        iou_matrix = box_iou(cls_p_boxes[:, :4], cls_g_boxes[:, :4])

        # 对每个 IoU 阈值独立进行匹配
        for i, iou_thresh in enumerate(iou_thresholds):
            # 记录当前阈值下，哪些 GT 已经被匹配了 (去重)
            gt_matched = torch.zeros(len(cls_g_boxes), dtype=torch.bool)
            
            # [核心循环] 遍历每个预测框 (cls_p_boxes 已经是按 Conf 降序排列的)
            for j in range(len(cls_p_boxes)):
                # 找到该预测框 P_i 与所有 GT 的 IoU
                ious = iou_matrix[j]
                
                # 筛选出: 1. IoU > 阈值  2. GT 尚未被匹配
                # 资料: "如果 GT_j 已经被别的预测框匹配过，则 P_i ... 算作 FP"
                candidates_mask = (ious > iou_thresh) & (~gt_matched)
                
                # 找到符合条件的 GT 索引
                candidate_gt_indices = torch.where(candidates_mask)[0]
                
                if len(candidate_gt_indices) > 0:
                    # 资料: "如果找到多个 GT_j，选择与 P_i 的 IoU 最大的那个"
                    # 获取这些候选 GT 的 IoU 值
                    candidate_ious = ious[candidates_mask]
                    # 找到最大 IoU 对应的索引 (在 candidates 中的相对位置)
                    max_iou_idx_local = torch.argmax(candidate_ious)
                    # 找到对应的真实 GT 索引
                    best_gt_idx = candidate_gt_indices[max_iou_idx_local]
                    
                    # 匹配成功 (TP)
                    # 找到 P_i 在全局 correct 矩阵中的位置
                    global_p_idx = p_indices_in_global[j]
                    correct[global_p_idx, i] = True
                    
                    # 标记该 GT 已被占用
                    gt_matched[best_gt_idx] = True

    # 返回时必须保证 conf 和 cls 是与 correct 对应的排序后顺序
    return correct, p_boxes[:, 4], p_boxes[:, 5], g_boxes[:, 4]

# def compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_thresholds):
#     """
#     计算单张图的 TP 矩阵 (物理坐标系)
#     """
#     # 2. 整理 Preds
#     p_boxes = []
#     for p in preds:
#         p_boxes.append([p['start_time'], p['start_frequency'], p['end_time'], p['end_frequency'], p['confidence'], p['class']])
#     p_boxes = torch.tensor(p_boxes) if len(p_boxes) > 0 else torch.zeros((0, 6))

#     # 3. 整理 GTs
#     g_boxes = []
#     for g in gts:
#         g_boxes.append([g['start_time'], g['start_frequency'], g['end_time'], g['end_frequency'], g['class']])
#     g_boxes = torch.tensor(g_boxes) if len(g_boxes) > 0 else torch.zeros((0, 5))

#     # 4. 初始化
#     correct = torch.zeros(len(p_boxes), len(iou_thresholds), dtype=torch.bool)
    
#     if len(p_boxes) == 0:
#         return correct, torch.tensor([]), torch.tensor([]), g_boxes[:, 4] if len(g_boxes) > 0 else torch.tensor([])

#     # 5. 按类别匹配
#     unique_classes = torch.unique(torch.cat([p_boxes[:, 5], g_boxes[:, 4]])) if len(g_boxes) > 0 else torch.unique(p_boxes[:, 5])

#     for cls in unique_classes:
#         p_mask = (p_boxes[:, 5] == cls)
#         g_mask = (g_boxes[:, 4] == cls)
        
#         cls_p_boxes = p_boxes[p_mask]
#         cls_g_boxes = g_boxes[g_mask]
#         p_indices = torch.where(p_mask)[0]

#         if len(cls_g_boxes) == 0 or len(cls_p_boxes) == 0:
#             continue

#         iou = box_iou(cls_g_boxes[:, :4], cls_p_boxes[:, :4]) 

#         for i, iou_thr in enumerate(iou_thresholds):
#             matches = torch.nonzero(iou >= iou_thr) 
#             if matches.shape[0] > 0:
#                 match_vals = iou[matches[:, 0], matches[:, 1]]
#                 matches = torch.cat([matches, match_vals[:, None]], 1)
#                 matches = matches[matches[:, 2].argsort(descending=True)]
#                 _, unique_p_idx = np.unique(matches[:, 1].cpu().numpy(), return_index=True)
#                 matches = matches[unique_p_idx]
#                 matches = matches[matches[:, 2].argsort(descending=True)]
#                 _, unique_g_idx = np.unique(matches[:, 0].cpu().numpy(), return_index=True)
#                 matches = matches[unique_g_idx]
                
#                 relative_pred_indices = matches[:, 1].long()
#                 absolute_pred_indices = p_indices[relative_pred_indices]
#                 correct[absolute_pred_indices, i] = True

#     return correct, p_boxes[:, 4], p_boxes[:, 5], g_boxes[:, 4]

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL, device=DEVICE)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    stats = [] 
    iou_v = torch.linspace(0.5, 0.95, 10)
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    print(f">>> Config: CONF_THRES={CONF_THRES}, IOU_THRES={IOU_THRES}")
    print(f">>> Visualization Limit: {VIS_LIMIT}")
    
    vis_count = 0 

    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        sample_dir = OUTPUT_DIR / bin_file.stem
        is_visualizing = (VIS_LIMIT == -1) or (vis_count < VIS_LIMIT)

        # 1. 运行预测 (V6 Inference System)
        preds, img_large, img_dims, meta, slices = system.predict(
            bin_file, json_file, 
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES      
        )
        
        # 2. 保存逻辑
        if is_visualizing:
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / f"{bin_file.stem}_large.jpg"), img_large)
            for img_slice, suffix in slices:
                cv2.imwrite(str(sample_dir / f"{bin_file.stem}{suffix}.jpg"), img_slice)
            system.save_results(preds, sample_dir / f"{bin_file.stem}_pred.json")
        else:
            system.save_results(preds, OUTPUT_DIR / f"{bin_file.stem}_pred.json")

        # 3. 计算指标
        gts = meta.get('signals', [])
        correct, conf, pcls, tcls = compute_tp_fp_per_image(preds, gts, meta, bin_file, iou_v)
        stats.append((correct.cpu(), conf.cpu(), pcls.cpu(), tcls.cpu()))
        
        # 4. 可视化
        if is_visualizing:
            h, w = img_dims[1], img_dims[0]
            obs = meta['observation_range']
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            iq_signal = raw_data[::2] + 1j * raw_data[1::2]
            fs = (obs[1] - obs[0]) * 1e6
            duration_ms = (len(iq_signal) / fs) * 1000
            
            # GT
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
            
            # Pred
            fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
            ax_pred.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
            for p in preds:
                x1, y1 = phys2pix_eval(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
                x2, y2 = phys2pix_eval(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
                color = 'red' if p['source'] == 'small' else 'orange'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax_pred.add_patch(rect)
                ax_pred.text(x1, y1-5, f"{p['class']} {p['confidence']:.2f}", color=color, fontsize=FONT_SIZE)
            plt.axis('off')
            plt.savefig(sample_dir / f"{bin_file.stem}_Pred.jpg")
            plt.close(fig_pred)

            vis_count += 1 

    # --- 最终指标计算 ---
    print("\n>>> Computing Final Metrics with Ultralytics...")
    if not stats:
        print("No stats collected.")
        return

    stats = [torch.cat(x, 0) for x in zip(*stats)]
    tp, conf, pcls, tcls = stats
    
    if tp.shape[0] > 0:
        names = {i: str(i) for i in range(14)}
        
        results = ap_per_class(
            tp.numpy(), 
            conf.numpy(), 
            pcls.numpy(), 
            tcls.numpy(), 
            plot=False, 
            names=names
        )
        
        p, r = results[2], results[3]
        ap_matrix = results[5]
        unique_classes = results[6]

        if ap_matrix.ndim == 1:
            ap_matrix = ap_matrix[None, :]

        ap50 = ap_matrix[:, 0]
        ap_mean = ap_matrix.mean(1)
        
        # 保存 TXT 结果
        result_txt_path = OUTPUT_DIR.with_suffix('.txt')
        
        def log_result(text, file_handle):
            print(text)
            file_handle.write(text + "\n")

        with open(result_txt_path, 'w', encoding='utf-8') as f:
            log_result("\n" + "="*90, f)
            log_result(f"CONF_THRES: {CONF_THRES} | IOU_THRES: {IOU_THRES}", f)
            log_result(f"{'Class':<10} | {'P':<10} | {'R':<10} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}", f)
            log_result("-" * 90, f)
            
            for i, c in enumerate(unique_classes):
                log_result(f"{int(c):<10} | {p[i]:.4f}     | {r[i]:.4f}     | {ap50[i]:.4f}     | {ap_mean[i]:.4f}", f)
            
            log_result("-" * 90, f)
            log_result(f"{'ALL':<10} | {p.mean():.4f}     | {r.mean():.4f}     | {ap50.mean():.4f}     | {ap_mean.mean():.4f}", f)
            log_result("=" * 90, f)
            
        print(f"\n[Success] Metrics saved to: {result_txt_path}")
    else:
        print("No valid predictions found.")

if __name__ == "__main__":
    main()