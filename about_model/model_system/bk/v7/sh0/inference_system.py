import numpy as np
import json
import cv2
import torch
import torchvision
import math
from scipy.signal import stft
from ultralytics import YOLO

# ==============================================================================
#                                 CONFIGURATION
# ==============================================================================

# [核心方案选择]
# 0: Base 方案 (单模型，预测所有类别)
# 1: Dual 方案 (双分支，Large预测宽带 + Small预测窄带并拼接)
SCHEME = 0

# --- System Defaults ---
OVERLAP_RATIO = 0.5
SLICE_SIZE = 640
SLICE_OVERLAP = 0.2

# ==================== SCHEME 0: Base Configuration ====================
# [说明] 这里的参数默认与 Large 一致，但独立设置，方便单独调优
SAVE_IMAGE_FORMAT_BASE = 'jpg'
STFT_MODE_BASE = 2           
FIXED_NPERSEG_BASE = 1024
FREQ_RES_KHZ_BASE = 20
USE_DB_SCALE_BASE = False
NORM_TYPE_BASE = 'SAMPLE'
GLOBAL_MIN_DB_BASE = -140.0
GLOBAL_MAX_DB_BASE = 30.0

# ==================== SCHEME 1: Dual Configuration ====================

# --- Large Branch ---
SAVE_IMAGE_FORMAT_LARGE = 'jpg'
STFT_MODE_LARGE = 2 
FIXED_NPERSEG_LARGE = 1024
FREQ_RES_KHZ_LARGE = 20
USE_DB_SCALE_LARGE = False
NORM_TYPE_LARGE = 'SAMPLE' 
GLOBAL_MIN_DB_LARGE = -140.0
GLOBAL_MAX_DB_LARGE = 30.0

# --- Small Branch ---
SAVE_IMAGE_FORMAT_SMALL = 'jpg'
# Small 分支强制 STFT_MODE=3 (固定频率分辨率) 以适配切片
FREQ_RES_KHZ_SMALL = 5
USE_DB_SCALE_SMALL = False
NORM_TYPE_SMALL = 'SAMPLE'
GLOBAL_MIN_DB_SMALL = -140.0
GLOBAL_MAX_DB_SMALL = 30.0

# --- Class Routing Logic ---
# Scheme 1 专用：定义哪些类走 Large，哪些走 Small
LARGE_IGNORED_CLASSES = [9, 12, 13] 
SMALL_TARGET_CLASSES = [9, 10, 12, 13]

# ==============================================================================
#                             INFERENCE SYSTEM
# ==============================================================================

class SignalInferenceSystem:
    def __init__(self, base_model_path=None, large_model_path=None, small_model_path=None, device='cuda:0'):
        print(f">>> Initializing Inference System (SCHEME={SCHEME})...")
        self.device = device
        self.scheme = SCHEME
        
        if self.scheme == 0:
            if base_model_path is None: raise ValueError("SCHEME=0 requires base_model_path")
            print(f">>> Loading Base Model: {base_model_path}")
            self.model_base = YOLO(base_model_path)
            
        elif self.scheme == 1:
            if large_model_path is None or small_model_path is None: 
                raise ValueError("SCHEME=1 requires large_model_path and small_model_path")
            print(f">>> Loading Large Model: {large_model_path}")
            self.model_large = YOLO(large_model_path)
            print(f">>> Loading Small Model: {small_model_path}")
            self.model_small = YOLO(small_model_path)
        else:
            raise ValueError(f"Unknown SCHEME: {SCHEME}")
            
        print(">>> Models Loaded.")

    def _process_stft_configurable(self, iq_signal, fs, stft_mode, fixed_nperseg, freq_res_khz, use_db_scale, norm_type, global_min_db, global_max_db):
        signal_len = len(iq_signal)
        nperseg = 256
        
        if stft_mode == 1:
            nperseg = fixed_nperseg
            if nperseg > signal_len: nperseg = signal_len
        elif stft_mode == 2:
            try:
                calculated_nperseg = int(math.sqrt(signal_len / (1 - OVERLAP_RATIO)))
                if calculated_nperseg % 2 != 0: calculated_nperseg += 1
                nperseg = min(calculated_nperseg, signal_len)
                nperseg = max(nperseg, 64)
            except ValueError: nperseg = 256
        elif stft_mode == 3:
            nperseg = int(fs / (freq_res_khz * 1000))
            if nperseg > signal_len: nperseg = signal_len
            if nperseg < 64: nperseg = 64

        noverlap = int(nperseg * OVERLAP_RATIO)
        f, t, Zxx = stft(iq_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        magnitude = np.abs(Zxx)
        
        if use_db_scale:
            data = 20 * np.log10(magnitude + 1e-12)
        else:
            data = magnitude

        if norm_type == 'GLOBAL':
            data = np.clip(data, global_min_db, global_max_db)
            data = (data - global_min_db) / (global_max_db - global_min_db)
        else:
            local_min, local_max = data.min(), data.max()
            if local_max > local_min:
                data = (data - local_min) / (local_max - local_min)
            else:
                data = np.zeros_like(data)
            
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
        if isinstance(data_obj, torch.Tensor): return data_obj.cpu().numpy()
        elif isinstance(data_obj, np.ndarray): return data_obj
        return np.array(data_obj)

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

    def predict(self, bin_path, json_path, conf_thres=0.001, iou_thres=0.7):
        with open(json_path, 'r') as f: meta = json.load(f)
        obs_range = meta['observation_range']
        f_min, f_max = obs_range[0], obs_range[1]
        bw_mhz = f_max - f_min
        fs = bw_mhz * 1e6
        raw_data = np.fromfile(bin_path, dtype=np.float16)
        iq_signal = raw_data[::2] + 1j * raw_data[1::2]
        duration = len(iq_signal) / fs

        predictions = []
        debug_slices = []
        # 用于可视化的主图
        img_vis = None 
        # 用于坐标转换的尺寸
        h_vis, w_vis = 0, 0

        # Helper: Pixel to Physical
        def pixel_to_physical(px_box, img_w, img_h):
            x1, y1, x2, y2 = px_box
            t_start = (x1 / img_w) * duration * 1000 
            t_end   = (x2 / img_w) * duration * 1000 
            freq_start = (y1 / img_h) * bw_mhz + f_min
            freq_end   = (y2 / img_h) * bw_mhz + f_min
            return t_start, t_end, freq_start, freq_end

        # ================= SCHEME 0: Base (Single Model) =================
        if self.scheme == 0:
            img_base, shape_base = self._process_stft_configurable(
                iq_signal, fs, STFT_MODE_BASE, FIXED_NPERSEG_BASE, FREQ_RES_KHZ_BASE, 
                USE_DB_SCALE_BASE, NORM_TYPE_BASE, GLOBAL_MIN_DB_BASE, GLOBAL_MAX_DB_BASE
            )
            h_vis, w_vis = shape_base[0], shape_base[1]
            img_vis = img_base
            
            res_base = self.model_base.predict(
                img_base, conf=conf_thres, iou=iou_thres, device=self.device, verbose=False, imgsz=640, rect=False
            )[0]
            
            if res_base.boxes is not None and res_base.boxes.data is not None:
                boxes_np = self._safe_to_numpy(res_base.boxes.data)
                for box in boxes_np:
                    cls_id = int(box[5])
                    ts, te, fs, fe = pixel_to_physical(box[:4], w_vis, h_vis)
                    predictions.append({
                        "class": cls_id, "start_time": float(min(ts, te)), "end_time": float(max(ts, te)), 
                        "start_frequency": float(min(fs, fe)), "end_frequency": float(max(fs, fe)), 
                        "confidence": float(box[4]), "source": "base"
                    })

        # ================= SCHEME 1: Dual (Large + Small) =================
        elif self.scheme == 1:
            # 1. Large Branch
            img_large, shape_large = self._process_stft_configurable(
                iq_signal, fs, STFT_MODE_LARGE, FIXED_NPERSEG_LARGE, FREQ_RES_KHZ_LARGE, 
                USE_DB_SCALE_LARGE, NORM_TYPE_LARGE, GLOBAL_MIN_DB_LARGE, GLOBAL_MAX_DB_LARGE
            )
            h_L, w_L = shape_large[0], shape_large[1]
            img_vis = img_large # 默认返回 Large 图用于可视化
            h_vis, w_vis = h_L, w_L
            
            res_large = self.model_large.predict(
                img_large, conf=conf_thres, iou=iou_thres, device=self.device, verbose=False, imgsz=640, rect=False
            )[0]
            
            if res_large.boxes is not None and res_large.boxes.data is not None:
                boxes_np = self._safe_to_numpy(res_large.boxes.data)
                for box in boxes_np:
                    cls_id = int(box[5])
                    if cls_id not in LARGE_IGNORED_CLASSES: 
                        ts, te, fs, fe = pixel_to_physical(box[:4], w_L, h_L)
                        predictions.append({
                            "class": cls_id, "start_time": float(min(ts, te)), "end_time": float(max(ts, te)), 
                            "start_frequency": float(min(fs, fe)), "end_frequency": float(max(fs, fe)), 
                            "confidence": float(box[4]), "source": "large"
                        })
            
            # 2. Small Branch
            img_small_full, shape_small = self._process_stft_configurable(
                iq_signal, fs, 3, 0, FREQ_RES_KHZ_SMALL, 
                USE_DB_SCALE_SMALL, NORM_TYPE_SMALL, GLOBAL_MIN_DB_SMALL, GLOBAL_MAX_DB_SMALL
            )
            h_S, w_S = shape_small[0], shape_small[1]
            
            slice_coords = self._get_slice_coords(h_S, w_S)
            small_boxes_global = [] 
            
            for i, (sy1, sy2, sx1, sx2) in enumerate(slice_coords):
                img_slice = img_small_full[sy1:sy2, sx1:sx2]
                debug_slices.append((img_slice, f"_s{i}"))
                res_slice = self.model_small.predict(
                    img_slice, conf=conf_thres, iou=iou_thres, device=self.device, verbose=False, imgsz=640, rect=False
                )[0]
                
                if res_slice.boxes is not None and res_slice.boxes.data is not None:
                    boxes_slice = self._safe_to_numpy(res_slice.boxes.data)
                    for b in boxes_slice:
                        gx1, gy1, gx2, gy2 = b[0]+sx1, b[1]+sy1, b[2]+sx1, b[3]+sy1
                        small_boxes_global.append([gx1, gy1, gx2, gy2, b[4], b[5]])

            if len(small_boxes_global) > 0:
                small_boxes_np = np.array(small_boxes_global)
                stitched_h = self._stitch_1d(small_boxes_np, main_axis_idx=0, cross_axis_idx=1)
                stitched_hv = self._stitch_1d(stitched_h, main_axis_idx=1, cross_axis_idx=0) if len(stitched_h) > 0 else np.array([])
                if len(stitched_hv) > 0:
                    small_tensor = torch.tensor(stitched_hv, device=self.device)
                    keep_indices = torchvision.ops.nms(small_tensor[:, :4], small_tensor[:, 4], 0.3)
                    final_small_preds = small_tensor[keep_indices].cpu().numpy()
                    
                    for box in final_small_preds:
                        cls_id = int(box[5])
                        if cls_id in SMALL_TARGET_CLASSES: 
                            ts, te, fs, fe = pixel_to_physical(box[:4], w_S, h_S)
                            predictions.append({
                                "class": cls_id, "start_time": float(min(ts, te)), "end_time": float(max(ts, te)), 
                                "start_frequency": float(min(fs, fe)), "end_frequency": float(max(fs, fe)), 
                                "confidence": float(box[4]), "source": "small"
                            })

        return predictions, img_vis, (w_vis, h_vis), meta, debug_slices

    def save_results(self, predictions, output_json_path):
        output = {"signals": predictions}
        with open(output_json_path, 'w') as f: json.dump(output, f, indent=4)