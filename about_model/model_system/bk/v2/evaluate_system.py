import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from inference_system import SignalInferenceSystem

# ================= 配置 =================
TEST_DIR = Path(r'E:\huangwenhao\processed_datasets\dataset_test')  # Test集路径
OUTPUT_DIR = Path(r'E:\huangwenhao\test_results\test_results_v2_2') # 结果保存路径
LARGE_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_large_jpg_1\weights\best.pt'
SMALL_MODEL = r'E:\huangwenhao\runs\v8\train\train_v8_small_jpg_1_epochs40\weights\best.pt'
DEVICE = 'cuda:0'
FONT_SIZE = 8

class COCOStyleEvaluator:
    # ... (保持之前的 Evaluator 类代码不变，为了节省篇幅这里省略，请直接复用之前的类定义) ...
    # 只需要复制 calculate_iou, process_batch, _compute_ap, compute 这几个方法
    def __init__(self, iou_thresholds=np.linspace(0.5, 0.95, 10)):
        self.iou_thresholds = iou_thresholds
        self.stats = []
        self.all_classes = set()
    
    def calculate_iou(self, box1, box2):
        t_min, t_max = max(box1[0], box2[0]), min(box1[1], box2[1])
        f_min, f_max = max(box1[2], box2[2]), min(box1[3], box2[3])
        if t_max <= t_min or f_max <= f_min: return 0.0
        inter_area = (t_max - t_min) * (f_max - f_min)
        area1 = (box1[1] - box1[0]) * (box1[3] - box1[2])
        area2 = (box2[1] - box2[0]) * (box2[3] - box2[2])
        return inter_area / (area1 + area2 - inter_area + 1e-16)

    def process_batch(self, preds, gts):
        pred_boxes, pred_scores, pred_cls = [], [], []
        for p in preds:
            pred_boxes.append([p['start_time'], p['end_time'], p['start_frequency'], p['end_frequency']])
            pred_scores.append(p['confidence'])
            pred_cls.append(p['class'])
            self.all_classes.add(p['class'])
        gt_boxes, gt_cls = [], []
        for g in gts:
            gt_boxes.append([g['start_time'], g['end_time'], g['start_frequency'], g['end_frequency']])
            gt_cls.append(g['class'])
            self.all_classes.add(g['class'])
        self.stats.append((np.array(pred_boxes), np.array(pred_scores), np.array(pred_cls), np.array(gt_boxes), np.array(gt_cls)))

    def _compute_ap(self, recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    def compute(self):
        sorted_classes = sorted(list(self.all_classes))
        metrics_per_class = {}
        for cls_id in sorted_classes:
            targets_cls = 0
            all_preds_cls = []
            for (p_boxes, p_scores, p_cls, g_boxes, g_cls) in self.stats:
                curr_gts = g_boxes[g_cls == cls_id]
                targets_cls += len(curr_gts)
                mask = (p_cls == cls_id)
                curr_preds, curr_scores = p_boxes[mask], p_scores[mask]
                if len(curr_preds) == 0: continue
                
                iou_mat = np.zeros((len(curr_preds), len(curr_gts))) if len(curr_gts) > 0 else np.zeros((len(curr_preds), 0))
                for i, pb in enumerate(curr_preds):
                    for j, gb in enumerate(curr_gts):
                        iou_mat[i, j] = self.calculate_iou(pb, gb)
                
                correct = np.zeros((len(curr_preds), len(self.iou_thresholds)), dtype=bool)
                if len(curr_gts) > 0:
                    for k, iou_thr in enumerate(self.iou_thresholds):
                        matches = np.where(iou_mat >= iou_thr)
                        if matches[0].size > 0:
                            matches = np.column_stack(matches)
                            matches = matches[np.argsort(-iou_mat[matches[:,0], matches[:,1]])]
                            gt_seen, pred_seen = set(), set()
                            for p_idx, g_idx in matches:
                                if p_idx in pred_seen or g_idx in gt_seen: continue
                                correct[p_idx, k] = True
                                gt_seen.add(g_idx)
                                pred_seen.add(p_idx)
                for i in range(len(curr_preds)): all_preds_cls.append((curr_scores[i], correct[i]))

            if targets_cls == 0 and len(all_preds_cls) == 0: continue
            if targets_cls == 0 or len(all_preds_cls) == 0:
                metrics_per_class[cls_id] = {'ap50': 0.0, 'ap5095': 0.0, 'p': 0.0, 'r': 0.0, 'nt': targets_cls}
                continue
            
            all_preds_cls.sort(key=lambda x: x[0], reverse=True)
            correct_mat = np.array([x[1] for x in all_preds_cls])
            aps = []
            for k in range(len(self.iou_thresholds)):
                tps = correct_mat[:, k]
                aps.append(self._compute_ap(np.cumsum(tps)/(targets_cls+1e-16), np.cumsum(tps)/(np.cumsum(tps)+np.cumsum(~tps)+1e-16)))
            
            tp_sum = np.sum(correct_mat[:, 0])
            metrics_per_class[cls_id] = {'ap50': aps[0], 'ap5095': np.mean(aps), 
                                         'p': tp_sum/(len(correct_mat)+1e-16), 
                                         'r': tp_sum/(targets_cls+1e-16), 'nt': targets_cls}
        return metrics_per_class

def phys2pix(val_t, val_f, duration_ms, f_range, img_w, img_h):
    x = (val_t / duration_ms) * img_w
    y = ((val_f - f_range[0]) / (f_range[1] - f_range[0])) * img_h
    return x, y

def main():
    system = SignalInferenceSystem(LARGE_MODEL, SMALL_MODEL, DEVICE)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR) # 清空旧结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bin_files = list(TEST_DIR.glob('*.bin'))

    # # 如果文件太多，可以先测试 10 个
    # bin_files = bin_files[:10]

    evaluator = COCOStyleEvaluator()
    
    print(f">>> Start Testing on {len(bin_files)} files...")
    
    for bin_file in tqdm(bin_files):
        json_file = bin_file.with_suffix('.json')
        if not json_file.exists(): continue
        
        # 创建每个样本的专属文件夹
        sample_dir = OUTPUT_DIR / bin_file.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 运行预测 (接收额外的 debug_slices)
        preds, img_large, img_dims, meta, slices = system.predict(bin_file, json_file)
        
        # 2. 保存 Large 预处理图
        cv2.imwrite(str(sample_dir / f"{bin_file.stem}_large.jpg"), img_large)
        
        # 3. 保存 Small 切片图
        for img_slice, suffix in slices:
            cv2.imwrite(str(sample_dir / f"{bin_file.stem}{suffix}.jpg"), img_slice)
            
        # 4. 统计
        gts = meta.get('signals', [])
        evaluator.process_batch(preds, gts)
        
        # 5. 可视化 (GT 和 Pred)
        h, w = img_dims[1], img_dims[0]
        obs = meta['observation_range']
        raw_data = np.fromfile(bin_file, dtype=np.float16)
        iq_signal = raw_data[::2] + 1j * raw_data[1::2]
        fs = (obs[1] - obs[0]) * 1e6
        duration_ms = (len(iq_signal) / fs) * 1000
        
        # 画 GT
        fig_gt, ax_gt = plt.subplots(figsize=(12, 12 * h / w))
        ax_gt.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
        ax_gt.set_title(f"Ground Truth", fontsize=14, color='green')
        for g in gts:
            x1, y1 = phys2pix(g['start_time'], g['start_frequency'], duration_ms, obs, w, h)
            x2, y2 = phys2pix(g['end_time'], g['end_frequency'], duration_ms, obs, w, h)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax_gt.add_patch(rect)
            ax_gt.text(x1, y1-5, f"C{g['class']}", color='lime', fontsize=FONT_SIZE, fontweight='bold')
        plt.axis('off'); plt.tight_layout()
        plt.savefig(sample_dir / f"{bin_file.stem}_GT.jpg")
        plt.close(fig_gt)
        
        # 画 Pred
        fig_pred, ax_pred = plt.subplots(figsize=(12, 12 * h / w))
        ax_pred.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
        ax_pred.set_title(f"Predictions", fontsize=14, color='red')
        for p in preds:
            x1, y1 = phys2pix(p['start_time'], p['start_frequency'], duration_ms, obs, w, h)
            x2, y2 = phys2pix(p['end_time'], p['end_frequency'], duration_ms, obs, w, h)
            color = 'red' if p['source'] == 'small' else 'orange'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(x1, y1-5, f"C{p['class']} {p['confidence']:.2f}", color=color, fontsize=FONT_SIZE, fontweight='bold')
        plt.axis('off'); plt.tight_layout()
        plt.savefig(sample_dir / f"{bin_file.stem}_Pred.jpg")
        plt.close(fig_pred)
        
        # 保存 JSON
        system.save_results(preds, sample_dir / f"{bin_file.stem}_pred.json")

    # 6. 打印指标
    print("\n>>> Computing Final Metrics (COCO Style)...")
    results = evaluator.compute()
    print("\n" + "="*85)
    print(f"{'Class':<10} | {'Targets':<8} | {'P':<8} | {'R':<8} | {'mAP@.50':<10} | {'mAP@.5-.95':<12}")
    print("-" * 85)
    total_nt, map50_sum, map5095_sum, p_sum, r_sum, count = 0, 0, 0, 0, 0, 0
    for cls_id, res in results.items():
        print(f"{cls_id:<10} | {res['nt']:<8} | {res['p']:.4f}   | {res['r']:.4f}   | {res['ap50']:.4f}     | {res['ap5095']:.4f}")
        if res['nt'] > 0:
            total_nt += res['nt']; map50_sum += res['ap50']; map5095_sum += res['ap5095']; p_sum += res['p']; r_sum += res['r']; count += 1
    print("-" * 85)
    if count > 0:
        print(f"{'ALL':<10} | {total_nt:<8} | {p_sum/count:.4f}   | {r_sum/count:.4f}   | {map50_sum/count:.4f}     | {map5095_sum/count:.4f}")
    print("=" * 85)

if __name__ == "__main__":
    main()