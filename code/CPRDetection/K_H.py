import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import os
import cv2

# 封装卡尔曼滤波器初始化和更新逻辑
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)
        self.kf.R *= 10
        self.kf.P *= 10
        self.kf.Q *= 0.1
        self.time_since_update = 0
        self.history = []
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.label = None  # 初始化标签为空

    def predict(self):
        self.kf.predict()
        return self.convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.time_since_update = 0

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / h
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        x1 = x[0] - w / 2.0
        y1 = x[1] - h / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + h / 2.0
        return np.array([x1, y1, x2, y2])


# 封装匈牙利算法
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2)), np.arange(len(detections)), np.empty((0, 4))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, tracker in enumerate(trackers):
            predicted_bbox = tracker.predict().ravel()
            iou_matrix[d, t] = iou(det, predicted_bbox)


    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_indices = np.array(list(zip(row_ind, col_ind)))
    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] >= iou_threshold:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    return np.concatenate(matches, axis=0) if matches else np.empty((0, 2)), np.array(unmatched_detections), np.array(
        unmatched_trackers)


# 计算IOU的函数
def iou(bbox1, bbox2):
    x1, y1, x2, y2 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter_area / (bbox1_area + bbox2_area - inter_area)


# 封装原始代码的跟踪函数
trackers = []  # 确保 trackers 是全局的
def run_kalman_tracking(detections):
    global trackers  # 使用全局 trackers
    if len(trackers) == 0:
        # 第一次运行时，直接初始化跟踪器
        for det in detections:
            tracker = KalmanBoxTracker(det)
            trackers.append(tracker)

    current_detections = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
    labels = np.array([det[4] for det in detections])  # 提取每个检测的label

    # 匹配检测框与跟踪器
    matched_indices, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(current_detections, trackers)

    # 更新卡尔曼滤波器并设置对应的label
    for m in matched_indices:
        tracker = trackers[m[1]]
        tracker.update(current_detections[m[0]])
        tracker.label = labels[m[0]]  # 将检测框的label与跟踪器关联

    # 对于未匹配的检测，创建新的跟踪器，并设置初始的label
    for i in unmatched_detections:
        tracker = KalmanBoxTracker(current_detections[i])
        tracker.label = labels[i]  # 关联初始label
        trackers.append(tracker)

    # 最终的检测结果 (经过卡尔曼滤波器和匈牙利算法后的预测)
    final_detections = []
    for tracker in trackers:
        final_detections.append(tracker.predict().ravel())

    return final_detections, trackers, matched_indices


# 设置输出目录
output_base_dir = "./output"  # 根据你的路径设置
yolo_kh_lab_output_dir = os.path.join(output_base_dir, "kh-yolo-lab")
if not os.path.exists(yolo_kh_lab_output_dir):
    os.makedirs(yolo_kh_lab_output_dir)


