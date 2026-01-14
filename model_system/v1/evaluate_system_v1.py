import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from inference_system_v1 import SignalInferenceSystem

# ================= 配置 =================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test')  # Test集路径
OUTPUT_DIR = Path(r'E:\huangwenhao\project-ver-1-20260107\model_system\test_results') # 结果保存路径
LARGE_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_large_jpg_1\weights\best.pt'
SMALL_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_small_jpg_1_epochs40\weights\best.pt'
DEVICE = 'cuda:0'

# 绘图配置
DRAW_GT_SEPARATELY = True  # 是否分开画图
FONT_SIZE = 8

class COCOStyleEvaluator:
    """
    完全对标 COCO/YOLO 标准的评估器
    计算 mAP50, mAP50-95, Precision, Recall
    """
    def __init__(self, iou_thresholds=np.linspace(0.5, 0.95, 10)):
        self.iou_thresholds = iou_thresholds
        self.stats = []  # 存储每张图的统计信息
        self.all_classes = set()

    def calculate_iou(self, box1, box2):
        """计算物理坐标 IoU [t1, t2, f1, f2]"""
        # Intersection
        t_min = max(box1[0], box2[0])
        t_max = min(box1[1], box2[1])
        f_min = max(box1[2], box2[2])
        f_max = min(box1[3], box2[3])
        
        if t_max <= t_min or f_max <= f_min:
            return 0.0
            
        inter_area = (t_max - t_min) * (f_max - f_min)
        area1 = (box1[1] - box1[0]) * (box1[3] - box1[2])
        area2 = (box2[1] - box2[0]) * (box2[3] - box2[2])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-16)

    def process_batch(self, preds, gts):
        """
        处理单张图片的预测结果
        preds: list of dict {'class', 'confidence', 'start_time', ...}
        gts: list of dict {'class', 'start_time', ...}
        """
        # 统一格式化为 [t1, t2, f1, f2]
        pred_boxes = []
        pred_scores = []
        pred_cls = []
        for p in preds:
            pred_boxes.append([p['start_time'], p['end_time'], p['start_frequency'], p['end_frequency']])
            pred_scores.append(p['confidence'])
            pred_cls.append(p['class'])
            self.all_classes.add(p['class'])
            
        gt_boxes = []
        gt_cls = []
        for g in gts:
            gt_boxes.append([g['start_time'], g['end_time'], g['start_frequency'], g['end_frequency']])
            gt_cls.append(g['class'])
            self.all_classes.add(g['class'])
            
        # 转换为 Numpy
        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)
        pred_cls = np.array(pred_cls)
        gt_boxes = np.array(gt_boxes)
        gt_cls = np.array(gt_cls)
        
        self.stats.append((pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls))

    def _compute_ap(self, recall, precision):
        """计算单类别的 AP (All-point interpolation)"""
        # 在开头和结尾插入哨兵值
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # 计算包络线 (Envelope)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 计算面积 (AUC)
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute(self):
        """计算最终指标"""
        # 整理所有类别
        sorted_classes = sorted(list(self.all_classes))
        
        # 结果存储
        metrics_per_class = {} # {cls_id: {'ap50': x, 'ap5095': x, 'p': x, 'r': x}}
        
        # 遍历每个类别
        for cls_id in sorted_classes:
            targets_cls = 0
            # 收集该类别的所有预测和所有GT
            # 为了计算 AP，我们需要把整个测试集该类别的所有预测框拉在一起排序
            all_preds_cls = [] # (score, tp_per_iou_threshold)
            
            for (p_boxes, p_scores, p_cls, g_boxes, g_cls) in self.stats:
                # 筛选当前类别的 GT
                curr_gts = g_boxes[g_cls == cls_id]
                targets_cls += len(curr_gts)
                
                # 筛选当前类别的 Pred
                mask = (p_cls == cls_id)
                curr_preds = p_boxes[mask]
                curr_scores = p_scores[mask]
                
                if len(curr_preds) == 0:
                    continue
                
                # 计算 IoU 矩阵 [Num_Preds, Num_GTs]
                if len(curr_gts) > 0:
                    iou_mat = np.zeros((len(curr_preds), len(curr_gts)))
                    for i, pb in enumerate(curr_preds):
                        for j, gb in enumerate(curr_gts):
                            iou_mat[i, j] = self.calculate_iou(pb, gb)
                else:
                    iou_mat = np.zeros((len(curr_preds), 0))
                
                # 针对每个 IoU 阈值判断 TP/FP
                # correct: [Num_Preds, 10] (bool matrix)
                correct = np.zeros((len(curr_preds), len(self.iou_thresholds)), dtype=bool)
                
                if len(curr_gts) > 0:
                    # 对每个 IoU 阈值
                    for k, iou_thr in enumerate(self.iou_thresholds):
                        # 贪心匹配：优先匹配 IoU 高的
                        # 这里的逻辑比较简化，标准COCO会按分数排序全局匹配
                        # 但针对单图匹配也是常见的做法
                        matches = np.where(iou_mat >= iou_thr)
                        if matches[0].size > 0:
                            matches = np.column_stack(matches) # [[pidx, gidx], ...]
                            # 按 IoU 降序排列匹配对
                            match_ious = iou_mat[matches[:,0], matches[:,1]]
                            matches = matches[np.argsort(-match_ious)]
                            
                            gt_seen = set()
                            pred_seen = set()
                            
                            for p_idx, g_idx in matches:
                                if p_idx in pred_seen or g_idx in gt_seen:
                                    continue
                                correct[p_idx, k] = True
                                gt_seen.add(g_idx)
                                pred_seen.add(p_idx)
                                
                # 记录 (score, correct_vector)
                for i in range(len(curr_preds)):
                    all_preds_cls.append((curr_scores[i], correct[i]))

            # 如果没有 GT 也没有 Pred
            if targets_cls == 0 and len(all_preds_cls) == 0:
                continue
                
            # 如果有 Pred 没 GT -> AP=0
            if targets_cls == 0:
                metrics_per_class[cls_id] = {'ap50': 0.0, 'ap5095': 0.0, 'p': 0.0, 'r': 0.0, 'nt': 0}
                continue
                
            # 如果有 GT 没 Pred -> AP=0
            if len(all_preds_cls) == 0:
                metrics_per_class[cls_id] = {'ap50': 0.0, 'ap5095': 0.0, 'p': 0.0, 'r': 0.0, 'nt': targets_cls}
                continue

            # --- 计算指标 ---
            # 1. 按置信度降序排列
            all_preds_cls.sort(key=lambda x: x[0], reverse=True)
            
            # [Num_All_Preds, 10]
            correct_mat = np.array([x[1] for x in all_preds_cls]) 
            
            # 计算 AP @ 每个阈值
            aps = []
            for k in range(len(self.iou_thresholds)):
                tps = correct_mat[:, k]
                fps = ~tps
                
                tp_cumsum = np.cumsum(tps)
                fp_cumsum = np.cumsum(fps)
                
                recalls = tp_cumsum / (targets_cls + 1e-16)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                aps.append(self._compute_ap(recalls, precisions))
            
            # 计算最佳 P 和 R (通常取 IoU=0.5 下 F1-score 最大的点，或者直接取置信度 0.001 处的)
            # 这里简化：取 IoU=0.5 时最后一点的 Recall 和 Precision (近似)
            # 更严格的做法是找 F1-Max 点，这里为了展示方便：
            tp_05 = correct_mat[:, 0]
            tp_sum = np.sum(tp_05)
            fp_sum = len(tp_05) - tp_sum
            
            # 记录结果
            metrics_per_class[cls_id] = {
                'ap50': aps[0],                 # IoU=0.5
                'ap5095': np.mean(aps),         # Average over 0.5:0.95
                'p': tp_sum / (tp_sum + fp_sum + 1e-16), # Global Precision @ all preds
                'r': tp_sum / (targets_cls + 1e-16),     # Global Recall @ all preds
                'nt': targets_cls
            }

        return metrics_per_class

def visualize_separated(img_rgb, img_dims, gts, preds, obs_range, output_path_base):
    """分开画 GT 和 Pred"""
    h, w = img_dims[1], img_dims[0]
    duration = len(img_rgb[0]) # 这里的 img_rgb 已经是像素了，不需要 duration 换算 pixel
    # 重新计算 duration 和 bw 方便画图
    # 注意：inference 里的 pixel_to_physical 是除以 w 乘 duration
    # 这里我们要反过来：physical_to_pixel
    # duration (ms) = 样本总时长
    # bw (MHz) = f_max - f_min
    
    # 获取 duration 的 trick: 我们没有直接传 duration 进来
    # 但是我们知道： x / w = t / duration_ms
    # 所以 x = t * w / duration_ms
    # 为了简单，我们不需要 duration，只需要知道 t 和 f 在 range 里的比例即可
    
    t_min, t_max = 0, 1000.0 # 假设归一化后的比例，或者直接利用 time range
    # 实际上，preds 和 gts 里的 time 是 ms，freq 是 MHz
    # 我们需要 obs_range 来确定 freq 的比例
    f_min, f_max = obs_range[0], obs_range[1]
    
    # 我们还需要该样本的总时长 duration_ms 来确定 time 的比例
    # inference_system 里没有直接把 duration_ms 传出来，
    # 我们可以通过 file size 算，或者为了方便，修改 infer system 返回 duration
    # 但这里最简单的办法是：我们在 evaluate_loop 里算一次传进来
    pass 

def phys2pix(val_t, val_f, duration_ms, f_range, img_w, img_h):
    """物理坐标 -> 像素坐标"""
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL, DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))
    # 如果文件太多，可以先测试 100 个
    # bin_files = bin_files[:100]
    
    evaluator = COCOStyleEvaluator()
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    print(f">>> Mode: Separated Visualization (GT vs Pred)")
    
    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        # 1. 运行系统预测
        preds, img_vis, img_dims, meta = system.predict(bin_file, json_file)
        
        # 2. 读取真值
        gts = meta.get('signals', [])
        
        # 3. 记录数据用于计算指标
        evaluator.process_batch(preds, gts)
        
        # --- 可视化部分 ---
        # 计算物理参数用于绘图
        h, w = img_dims[1], img_dims[0]
        obs = meta['observation_range'] # [f_min, f_max]
        raw_data = np.fromfile(bin_file, dtype=np.float16)
        iq_signal = raw_data[::2] + 1j * raw_data[1::2]
        fs = (obs[1] - obs[0]) * 1e6
        duration_ms = (len(iq_signal) / fs) * 1000
        
        # A. 画 GT 图
        fig_gt, ax_gt = plt.subplots(figsize=(12, 12 * h / w))
        ax_gt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)) # 背景
        ax_gt.set_title(f"{bin_file.stem} - Ground Truth", fontsize=14, color='green')
        
        for g in gts:
            x1, y1 = phys2pix(g['start_time'], g['start_frequency'], duration_ms, obs, w, h)
            x2, y2 = phys2pix(g['end_time'], g['end_frequency'], duration_ms, obs, w, h)
            # 修正坐标 (y轴可能反了? STFT通常低频在下，但在img中y=0在上)
            # inference_system 里 logic: freq_start = (y1 / img_h)... 
            # 这意味着 y=0 对应 f_min。如果 stft 做了 flipud 另说。
            # 这里按 inference_system 的逻辑：y 正比于 freq。
            # 画图时 Rect 需要 (x, y, w, h)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax_gt.add_patch(rect)
            ax_gt.text(x1, y1-5, f"C{g['class']}", color='lime', fontsize=FONT_SIZE, fontweight='bold')
            
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bin_file.stem}_GT.jpg")
        plt.close(fig_gt)
        
        # B. 画 Pred 图
        fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
        ax_pred.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        ax_pred.set_title(f"{bin_file.stem} - Predictions", fontsize=14, color='red')
        
        for p in preds:
            x1, y1 = phys2pix(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
            x2, y2 = phys2pix(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
            
            color = 'red' if p['source'] == 'small' else 'orange'
            label_text = f"C{p['class']} {p['confidence']:.2f}"
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(x1, y1-5, label_text, color=color, fontsize=FONT_SIZE, fontweight='bold')
            
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bin_file.stem}_Pred.jpg")
        plt.close(fig_pred)
        
        # 保存预测 JSON
        system.save_results(preds, OUTPUT_DIR / f"{bin_file.stem}_pred.json")

    # 4. 计算并打印最终指标
    print("\n>>> Computing Final Metrics (COCO Style)...")
    results = evaluator.compute()
    
    # 打印精美的表格
    print("\n" + "="*85)
    print(f"{'Class':<10} | {'Targets':<8} | {'P':<8} | {'R':<8} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}")
    print("-" * 85)
    
    total_nt = 0
    map50_sum = 0
    map5095_sum = 0
    p_sum = 0
    r_sum = 0
    count = 0
    
    for cls_id, res in results.items():
        print(f"{cls_id:<10} | {res['nt']:<8} | {res['p']:.4f}   | {res['r']:.4f}   | {res['ap50']:.4f}     | {res['ap5095']:.4f}")
        
        if res['nt'] > 0: # 只有存在 GT 的类才计入平均
            total_nt += res['nt']
            map50_sum += res['ap50']
            map5095_sum += res['ap5095']
            p_sum += res['p']
            r_sum += res['r']
            count += 1
            
    print("-" * 85)
    if count > 0:
        print(f"{'ALL':<10} | {total_nt:<8} | {p_sum/count:.4f}   | {r_sum/count:.4f}   | {map50_sum/count:.4f}     | {map5095_sum/count:.4f}")
    print("=" * 85)

if __name__ == "__main__":
    main()