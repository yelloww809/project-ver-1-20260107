import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from inference_system import SignalInferenceSystem
# [关键] 导入 Ultralytics 官方评估工具
from ultralytics.utils.metrics import ap_per_class

# ================= 配置 =================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test') 
OUTPUT_DIR = Path(r'E:\huangwenhao\test_results')
LARGE_MODEL = r'runs/detect/train_large/weights/best.pt'
SMALL_MODEL = r'runs/detect/train_small/weights/best.pt'
FONT_SIZE = 8

def phys2pix_for_eval(val_t, val_f, duration_ms, f_range, img_w, img_h):
    """物理坐标 -> 虚拟像素坐标 (用于统一尺度计算IoU)"""
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def process_batch_for_ultralytics(preds, gts, meta, bin_file):
    """
    将物理坐标转换为统一的 tensor 格式，供 ap_per_class 使用
    为了统一计算，我们将所有坐标映射到一个虚拟的 1000x1000 空间计算 IoU
    """
    # 获取物理参数
    obs = meta['observation_range']
    raw_data = np.fromfile(bin_file, dtype=np.float16)
    iq_signal = raw_data[::2] + 1j * raw_data[1::2]
    fs = (obs[1] - obs[0]) * 1e6
    duration_ms = (len(iq_signal) / fs) * 1000
    
    # 虚拟画布大小 (只要统一即可，不影响IoU比例)
    V_W, V_H = 1000.0, 1000.0

    # 1. 处理 GT
    target_cls = []
    target_bboxes = []
    for g in gts:
        target_cls.append(int(g['class']))
        x1, y1 = phys2pix_for_eval(g['start_time'], g['start_frequency'], duration_ms, obs, V_W, V_H)
        x2, y2 = phys2pix_for_eval(g['end_time'], g['end_frequency'], duration_ms, obs, V_W, V_H)
        target_bboxes.append([x1, y1, x2, y2])
    
    # 2. 处理 Preds
    pred_cls = []
    pred_bboxes = []
    pred_conf = []
    for p in preds:
        pred_cls.append(int(p['class']))
        pred_conf.append(float(p['confidence']))
        x1, y1 = phys2pix_for_eval(p['start_time'], p['start_frequency'], duration_ms, obs, V_W, V_H)
        x2, y2 = phys2pix_for_eval(p['end_time'], p['end_frequency'], duration_ms, obs, V_W, V_H)
        pred_bboxes.append([x1, y1, x2, y2])

    # 转换为 Tensor
    if len(target_bboxes) > 0:
        target_bboxes = torch.tensor(target_bboxes)
        target_cls = torch.tensor(target_cls)
    else:
        target_bboxes = torch.zeros((0, 4))
        target_cls = torch.zeros((0,))

    if len(pred_bboxes) > 0:
        pred_bboxes = torch.tensor(pred_bboxes)
        pred_conf = torch.tensor(pred_conf)
        pred_cls = torch.tensor(pred_cls)
    else:
        pred_bboxes = torch.zeros((0, 4))
        pred_conf = torch.zeros((0,))
        pred_cls = torch.zeros((0,))

    # 3. 计算 TP (使用 Ultralytics 的逻辑)
    # iou_thres array: 0.5, 0.55, ... 0.95
    iou_v = torch.linspace(0.5, 0.95, 10)
    
    if len(pred_bboxes) == 0:
        return np.zeros((0, 10), dtype=bool), np.array([]), np.array([]), target_cls.numpy()
    
    if len(target_bboxes) == 0:
        # 有预测没GT -> 全是FP
        tp = np.zeros((len(pred_bboxes), 10), dtype=bool)
        return tp, pred_conf.numpy(), pred_cls.numpy(), np.array([])

    # 计算 IoU Matrix
    # box_iou 是 Ultralytics 内部函数，这里我们手写一个简单的
    from ultralytics.utils.metrics import box_iou
    iou = box_iou(target_bboxes, pred_bboxes) # [M, N]
    
    # 计算 TP 矩阵 [N, 10]
    # 需要匹配 Ultralytics 的 match_predictions 逻辑
    # 为了简单，直接调用内部函数有点复杂，我们复用 ap_per_class 需要的输入：
    # ap_per_class 需要 tp 矩阵，我们需要自己算 batch match
    
    # 这里我们借用 Ultralytics 的 batch_probiou 或者简单的 match 逻辑
    # 鉴于环境配置，我们直接写一个简单的 match_predictions 函数，仿照 Ultralytics
    
    matches = match_predictions(pred_bboxes, target_bboxes, iou_v)
    # matches: [N, 10] bool
    
    return matches.cpu().numpy(), pred_conf.numpy(), pred_cls.numpy(), target_cls.numpy()

def match_predictions(pred_boxes, true_boxes, iou_thres):
    """
    仿照 Ultralytics match_predictions
    pred_boxes: [N, 4]
    true_boxes: [M, 4]
    iou_thres: [10]
    return: [N, 10] bool
    """
    from ultralytics.utils.metrics import box_iou
    iou = box_iou(true_boxes, pred_boxes)
    
    correct = torch.zeros(pred_boxes.shape[0], iou_thres.shape[0], dtype=torch.bool, device=pred_boxes.device)
    correct_class = torch.zeros_like(correct) # 我们在外面已经分了类，这里只做IoU匹配
    
    # 这里 ap_per_class 是按类计算的，所以我们这里只做纯 IoU 匹配
    # 假设传入的已经是同一类的 boxes (稍后在外面循环调用)
    # 或者我们直接对所有框做匹配，但 ap_per_class 内部会再次分类
    # Ultralytics 的 ap_per_class 需要的 tp 输入是针对 global list 的
    
    # 让我们换个思路：直接在 process_batch 里，对每张图，计算出每个 pred 的 TP 状态
    
    x = torch.where(iou >= iou_thres[0]) # 筛选至少满足 0.5 的
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    # 这部分逻辑稍微有点复杂，为了避免重复造轮子出错，建议：
    # 既然我们要用 ap_per_class，它只需要 (tp, conf, pred_cls, target_cls)
    # tp 是 [N_all, 10] 的 bool 矩阵。
    # 我们需要在每张图内部，计算该图的 pred 在 10 个阈值下是否匹配到了 GT
    
    return _compute_tp_per_image(pred_boxes, true_boxes, iou_thres)

def _compute_tp_per_image(pbox, tbox, iouv):
    """
    计算单张图片的 TP 矩阵 [N, 10]
    pbox: [N, 4]
    tbox: [M, 4]
    iouv: [10] (0.5 ... 0.95)
    """
    import torch
    from ultralytics.utils.metrics import box_iou
    
    ni = len(iouv)
    correct = np.zeros((len(pbox), ni), dtype=bool)
    
    if len(tbox) == 0:
        return correct
        
    iou = box_iou(tbox, pbox) # [M, N]
    
    # 对每个 IoU 阈值
    for i, threshold in enumerate(iouv):
        # 找到所有大于阈值的匹配 (gt_idx, pred_idx)
        matches = torch.nonzero(iou >= threshold) # [K, 2]
        if matches.shape[0] == 0:
            continue
            
        matches = matches.cpu().numpy()
        # 添加 IoU 值作为第三列 [gt_idx, pred_idx, iou_val]
        match_vals = iou[matches[:,0], matches[:,1]].cpu().numpy()
        matches = np.hstack([matches, match_vals[:, None]])
        
        # 贪心策略：按 IoU 降序
        # 1. Sort by IoU desc
        matches = matches[matches[:, 2].argsort()[::-1]]
        
        # 2. Unique Preds (每个 Pred 最多匹配一个 GT)
        # return_index=True 返回的是第一次出现的索引（因为已经降序，所以是最大的IoU）
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        
        # 3. Unique GTs (每个 GT 最多被一个 Pred 匹配)
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        # 记录 TP
        for m in matches:
            pred_idx = int(m[1])
            correct[pred_idx, i] = True
            
    return correct

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    
    # 统计全局数据，用于最后计算
    stats = [] # list of (tp, conf, pcls, tcls)
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    
    # [关键] 评估模式：极低置信度阈值
    EVAL_CONF = 0.001 
    EVAL_IOU = 0.6
    
    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        sample_dir = OUTPUT_DIR / bin_file.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 运行预测
        preds, img_large, img_dims, meta, slices = system.predict(
            bin_file, json_file, 
            conf_thres=EVAL_CONF, # 传入0.001
            iou_thres=EVAL_IOU
        )
        
        # 2. 保存图片 (为了不产生太多垃圾文件，可以注释掉下面几行，或者只保存前几个)
        # cv2.imwrite(str(sample_dir / f"{bin_file.stem}_large.jpg"), img_large)
        # ...
        
        # 3. 计算 TP/FP 状态 (针对 ap_per_class)
        gts = meta.get('signals', [])
        tp, conf, pcls, tcls = process_batch_for_ultralytics(preds, gts, meta, bin_file)
        
        stats.append((tp, conf, pcls, tcls))
        
        # 4. 可视化 (可选)
        # 注意：这里的 preds 包含大量低分框，画出来会全屏都是框
        # 如果要画图，建议用高分阈值再跑一次 system.predict，或者在画图时过滤 preds
        
        # 这里仅保存 JSON 用于调试
        system.save_results(preds, sample_dir / f"{bin_file.stem}_pred.json")

    # --- 计算最终指标 (使用 Ultralytics) ---
    print("\n>>> Computing Final Metrics using Ultralytics ap_per_class...")
    
    stats = [np.concatenate(x, 0) for x in zip(*stats)] 
    # tp: [N_all, 10], conf: [N_all], pcls: [N_all], tcls: [M_all]
    
    if len(stats) and stats[0].any():
        tp, conf, pcls, tcls = stats
        # ap_per_class 自动计算 P, R, AP, F1 等
        # plot=False 禁止画图 (我们需要matplotlib对象的话可以True)
        # names: 类别名称字典
        names = {i: str(i) for i in range(14)}
        
        results = ap_per_class(tp, conf, pcls, tcls, plot=False, names=names)
        
        # results: (tp, fp, p, r, f1, ap_unique_classes, ap)
        # ap: [N_cls, 10] -> mean over 10 gives mAP50-95, column 0 gives mAP50
        
        p, r, ap50, ap = results[2], results[3], results[6][:, 0], results[6].mean(1)
        
        print("\n" + "="*85)
        print(f"{'Class':<10} | {'P':<10} | {'R':<10} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}")
        print("-" * 85)
        
        # ap_unique_classes 是实际出现的类别索引
        unique_classes = results[5]
        
        for i, c in enumerate(unique_classes):
            print(f"{int(c):<10} | {p[i]:.4f}     | {r[i]:.4f}     | {ap50[i]:.4f}     | {ap[i]:.4f}")
            
        print("-" * 85)
        print(f"{'ALL':<10} | {p.mean():.4f}     | {r.mean():.4f}     | {ap50.mean():.4f}     | {ap.mean():.4f}")
        print("=" * 85)
        
    else:
        print("No detections found!")

if __name__ == "__main__":
    main()