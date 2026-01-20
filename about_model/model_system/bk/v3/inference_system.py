import numpy as np
import json
import cv2
import torch
import torchvision
import math
from scipy.signal import stft
from ultralytics import YOLO

# ================= 配置区域 =================
# [必须与 v8 训练完全一致]
GLOBAL_MIN_DB = -140.0
GLOBAL_MAX_DB = 30.0
FREQ_RES_LARGE = 20  # kHz
FREQ_RES_SMALL = 5   # kHz
OVERLAP_RATIO = 0.5
SLICE_SIZE = 640
SLICE_OVERLAP = 0.2

# 路由逻辑: 10类信号两边都测
LARGE_IGNORED_CLASSES = [9, 12, 13] 
SMALL_TARGET_CLASSES = [9, 10, 12, 13]

class SignalInferenceSystem:
    def __init__(self, large_model_path, small_model_path, device='cuda'):
        print(">>> Loading Models...")
        self.model_large = YOLO(large_model_path)
        self.model_small = YOLO(small_model_path)
        self.device = device
        print(">>> Models Loaded.")

    def _process_stft_v8_aligned(self, iq_signal, fs, freq_res_khz):
        """
        [完全复刻 v8_large_jpg_1 的预处理逻辑]
        """
        signal_len = len(iq_signal)
        
        # 1. 计算 nperseg (v8逻辑)
        # v8: nperseg = int(fs / (FREQ_RES_KHZ * 1000))
        nperseg = int(fs / (freq_res_khz * 1000))
        if nperseg > signal_len: nperseg = signal_len
        # v8 中有 max(nperseg, 64) 保护
        if nperseg < 64: nperseg = 64 
        
        noverlap = int(nperseg * OVERLAP_RATIO)
        
        # 2. STFT
        f, t, Zxx = stft(iq_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        magnitude = np.abs(Zxx)
        
        # 3. dB Scale
        data = 20 * np.log10(magnitude + 1e-12)
        
        # 4. Global Norm (v8逻辑)
        data = np.clip(data, GLOBAL_MIN_DB, GLOBAL_MAX_DB)
        data = (data - GLOBAL_MIN_DB) / (GLOBAL_MAX_DB - GLOBAL_MIN_DB)
        
        # 5. Convert to Image
        img_u8 = (data * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        return img_rgb, img_u8.shape

    def _get_slice_coords(self, img_h, img_w):
        stride_h = int(SLICE_SIZE * (1 - SLICE_OVERLAP))
        stride_w = int(SLICE_SIZE * (1 - SLICE_OVERLAP))
        coords = []
        for y in range(0, img_h, stride_h):
            y_end = min(y + SLICE_SIZE, img_h)
            if y >= img_h: continue
            for x in range(0, img_w, stride_w):
                x_end = min(x + SLICE_SIZE, img_w)
                if x >= img_w: continue
                coords.append((y, y_end, x, x_end))
                if x_end == img_w: break
            if y_end == img_h: break
        return coords

    def _safe_to_numpy(self, data_obj):
        if isinstance(data_obj, torch.Tensor):
            return data_obj.cpu().numpy()
        elif isinstance(data_obj, np.ndarray):
            return data_obj
        else:
            return np.array(data_obj)

    # ... (_calculate_1d_overlap 和 _stitch_1d 保持之前的版本不变，此处省略以节省篇幅) ...
    def _calculate_1d_overlap(self, range1, range2):
        inter_min = max(range1[0], range2[0])
        inter_max = min(range1[1], range2[1])
        if inter_max <= inter_min: return 0.0
        inter_len = inter_max - inter_min
        union_len = (range1[1] - range1[0]) + (range2[1] - range2[0]) - inter_len
        return inter_len / (union_len + 1e-16)

    def _stitch_1d(self, boxes, main_axis_idx=0, cross_axis_idx=1, gap_thresh=10, overlap_thresh=0.7):
        if len(boxes) == 0: return []
        boxes_by_cls = {}
        for b in boxes:
            cls_id = int(b[5])
            if cls_id not in boxes_by_cls: boxes_by_cls[cls_id] = []
            boxes_by_cls[cls_id].append(b)
        merged_boxes = []
        for cls_id, cls_boxes in boxes_by_cls.items():
            cls_boxes.sort(key=lambda x: x[main_axis_idx])
            if not cls_boxes: continue
            current_box = cls_boxes[0].copy()
            for i in range(1, len(cls_boxes)):
                next_box = cls_boxes[i]
                curr_main = [current_box[0], current_box[2]] if main_axis_idx == 0 else [current_box[1], current_box[3]]
                next_main = [next_box[0], next_box[2]]       if main_axis_idx == 0 else [next_box[1], next_box[3]]
                curr_cross = [current_box[1], current_box[3]] if main_axis_idx == 0 else [current_box[0], current_box[2]]
                next_cross = [next_box[1], next_box[3]]       if main_axis_idx == 0 else [next_box[0], next_box[2]]
                cross_iou = self._calculate_1d_overlap(curr_cross, next_cross)
                is_connected = (next_main[0] <= curr_main[1] + gap_thresh)
                if cross_iou > overlap_thresh and is_connected:
                    if main_axis_idx == 0:
                        current_box[2] = max(current_box[2], next_box[2])
                        current_box[0] = min(current_box[0], next_box[0])
                        current_box[1] = min(current_box[1], next_box[1])
                        current_box[3] = max(current_box[3], next_box[3])
                    else:
                        current_box[3] = max(current_box[3], next_box[3])
                        current_box[1] = min(current_box[1], next_box[1])
                        current_box[0] = min(current_box[0], next_box[0])
                        current_box[2] = max(current_box[2], next_box[2])
                    current_box[4] = max(current_box[4], next_box[4])
                else:
                    merged_boxes.append(current_box)
                    current_box = next_box.copy()
            merged_boxes.append(current_box)
        return np.array(merged_boxes)

    def predict(self, bin_path, json_path, conf_thres=0.001, iou_thres=0.6):
        """
        [关键修改]
        默认 conf_thres 设为 0.001 以匹配 model.val() 的 mAP 计算标准
        默认 iou_thres 设为 0.6
        """
        # 1. 读取数据
        with open(json_path, 'r') as f:
            meta = json.load(f)
        obs_range = meta['observation_range']
        f_min, f_max = obs_range[0], obs_range[1]
        bw_mhz = f_max - f_min
        fs = bw_mhz * 1e6
        
        raw_data = np.fromfile(bin_path, dtype=np.float16)
        iq_signal = raw_data[::2] + 1j * raw_data[1::2]
        duration = len(iq_signal) / fs

        # 2. 预处理 (使用复刻的 v8 逻辑)
        img_large, shape_large = self._process_stft_v8_aligned(iq_signal, fs, FREQ_RES_LARGE)
        h_L, w_L = shape_large[0], shape_large[1]
        
        img_small_full, shape_small = self._process_stft_v8_aligned(iq_signal, fs, FREQ_RES_SMALL)
        h_S, w_S = shape_small[0], shape_small[1]
        
        # 3. 推理 (Large) 
        # [关键] imgsz=640, rect=False, conf=0.001 (传参进来)
        res_large_list = self.model_large.predict(
            img_large, 
            conf=conf_thres, 
            iou=iou_thres, 
            device=self.device, 
            verbose=False, 
            imgsz=640,  
            rect=False 
        )
        res_large = res_large_list[0]
        
        # 4. 推理 (Small)
        slice_coords = self._get_slice_coords(h_S, w_S)
        small_boxes_global = [] 
        debug_slices = []
        
        for i, (sy1, sy2, sx1, sx2) in enumerate(slice_coords):
            img_slice = img_small_full[sy1:sy2, sx1:sx2]
            debug_slices.append((img_slice, f"_s{i}"))

            res_slice_list = self.model_small.predict(
                img_slice, 
                conf=conf_thres, 
                iou=iou_thres, 
                device=self.device, 
                verbose=False, 
                imgsz=640
            )
            res_slice = res_slice_list[0]
            
            if res_slice.boxes is not None:
                boxes_data = res_slice.boxes.data 
                if boxes_data is not None and len(boxes_data) > 0:
                    boxes_np = self._safe_to_numpy(boxes_data)
                    for b in boxes_np:
                        gx1, gy1, gx2, gy2 = b[0]+sx1, b[1]+sy1, b[2]+sx1, b[3]+sy1
                        small_boxes_global.append([gx1, gy1, gx2, gy2, b[4], b[5]])

        # 5. Small 合并
        final_small_preds = []
        if len(small_boxes_global) > 0:
            small_boxes_np = np.array(small_boxes_global)
            stitched_h = self._stitch_1d(small_boxes_np, main_axis_idx=0, cross_axis_idx=1, gap_thresh=10, overlap_thresh=0.7)
            if len(stitched_h) > 0:
                stitched_hv = self._stitch_1d(stitched_h, main_axis_idx=1, cross_axis_idx=0, gap_thresh=10, overlap_thresh=0.7)
            else:
                stitched_hv = np.array([])
            
            if len(stitched_hv) > 0:
                small_tensor = torch.tensor(stitched_hv, device=self.device)
                keep_indices = torchvision.ops.nms(small_tensor[:, :4], small_tensor[:, 4], 0.3)
                final_small_preds = small_tensor[keep_indices].cpu().numpy()

        # 6. 坐标逆归一化 + 格式化
        predictions = []

        def pixel_to_physical(px_box, img_w, img_h):
            x1, y1, x2, y2 = px_box
            t_start = (x1 / img_w) * duration * 1000 
            t_end   = (x2 / img_w) * duration * 1000 
            freq_start = (y1 / img_h) * bw_mhz + f_min
            freq_end   = (y2 / img_h) * bw_mhz + f_min
            return t_start, t_end, freq_start, freq_end

        # 处理 Large
        if res_large.boxes is not None:
            boxes_large_data = res_large.boxes.data
            if boxes_large_data is not None and len(boxes_large_data) > 0:
                boxes_large_np = self._safe_to_numpy(boxes_large_data)
                for box in boxes_large_np:
                    cls_id = int(box[5])
                    if cls_id not in LARGE_IGNORED_CLASSES: 
                        ts, te, fs, fe = pixel_to_physical(box[:4], w_L, h_L)
                        predictions.append({
                            "class": cls_id,
                            "start_time": float(min(ts, te)), 
                            "end_time": float(max(ts, te)),
                            "start_frequency": float(min(fs, fe)), 
                            "end_frequency": float(max(fs, fe)),
                            "confidence": float(box[4]),
                            "source": "large",
                            # [新增] 返回原始像素坐标用于IoU计算
                            "box_px": box[:4].tolist(),
                            "img_dims": (w_L, h_L)
                        })

        # 处理 Small
        for box in final_small_preds:
            cls_id = int(box[5])
            if cls_id in SMALL_TARGET_CLASSES: 
                ts, te, fs, fe = pixel_to_physical(box[:4], w_S, h_S)
                predictions.append({
                    "class": cls_id,
                    "start_time": float(min(ts, te)), 
                    "end_time": float(max(ts, te)),
                    "start_frequency": float(min(fs, fe)), 
                    "end_frequency": float(max(fs, fe)),
                    "confidence": float(box[4]),
                    "source": "small",
                    # [注意] 这里 Small 的像素坐标是相对于 Small STFT 的，不能直接和 Large 混用计算
                    # 计算 mAP 最好统一用物理坐标
                    "box_px": box[:4].tolist(),
                    "img_dims": (w_S, h_S)
                })
        
        return predictions, img_large, (w_L, h_L), meta, debug_slices

    def save_results(self, predictions, output_json_path):
        output = {"signals": predictions}
        with open(output_json_path, 'w') as f:
            json.dump(output, f, indent=4)