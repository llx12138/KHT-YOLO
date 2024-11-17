import os
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import cv2
import shutil
from Compare_s import generate_metric_scatter_plots
from EvaluatingIndicator1 import generate_evaluation_plots
from K_H import run_kalman_tracking
from datetime import datetime
#
# 选定模型并加载自定义数据集
model = YOLO("../yolov8n.pt")
# model = YOLO("best.pt")
data_path = "heart.yaml"  # 自定义数据集的yaml文件路径
model.train(data=data_path, epochs=100)  # 取消注释以训练模型
metrics = model.val()  # 取消注释进行验证
# 定义标签映射
label_map = {
    0: "Dummy",
    1: "D-head",
    2: "D-chest",
    3: "D-C-occlusion",
    4: "D-H-occlusion"
}

# 为每个标签分配一个固定颜色 (BGR格式)
label_colors = {
    0: (0, 255, 0),  # Dummy - 绿色
    1: (255, 0, 0),  # D-head - 蓝色
    2: (0, 0, 255),  # D-chest - 红色
    3: (255, 255, 0),  # D-C-occlusion - 青色
    4: (255, 0, 255)  # D-H-occlusion - 洋红色
}
# 定义输出路径
output_base_dir = "../data/iou/preimages-yolo-240-iou-n1"

# 创建输出目录
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# 定义需要预测的文件夹
valid_dir = "../datasets/preimages-yolo/valid"

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
        self.confidence = 0.0  # 添加置信度属性

    def predict(self):
        self.kf.predict()
        # 更新置信度，可以根据需要进行修改
        self.confidence = 1.0 / (1.0 + np.sqrt(np.sum(self.kf.P[:4, :4])))  # 使用协方差矩阵的某种形式来评估置信度
        return self.convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.confidence = 1.0 / (1.0 + np.sqrt(np.sum(self.kf.P[:4, :4])))  # 更新置信度

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


trackers = []
def iou(bbox1, bbox2):
    x1, y1, x2, y2 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter_area / (bbox1_area + bbox2_area - inter_area)
# 匈牙利算法    iou_threshold：容忍度
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2)), np.arange(len(detections)), np.empty((0, 4))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    # 使用跟踪器对象进行匹配，而非预测边界框的数组
    for d, det in enumerate(detections):
        for t, tracker in enumerate(trackers):
            predicted_bbox = tracker.predict().ravel()  # 展平为一维数组
            iou_matrix[d, t] = iou(det, predicted_bbox)  # 使用展平的预测框

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

def filter_nearby_boxes(matched_indices, detections, trackers, iou_threshold=0.5):
    filtered_indices = []

    # 统计当前图像中每个标签的数量
    label_counts = {}
    for det in detections:
        label = int(det[4])
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # 迭代所有匹配的检测框
    for i, idx1 in enumerate(matched_indices):
        bbox1 = trackers[idx1[1]].predict().ravel()  # 使用卡尔曼滤波器的预测框
        label1 = int(detections[idx1[0]][4])  # 标签
        conf1 = trackers[idx1[1]].confidence  # 获取卡尔曼滤波器的置信度

        keep = True
        # 检查当前检测框是否与之前的检测框有高 IOU 重叠
        for idx2 in filtered_indices:
            bbox2 = trackers[idx2[1]].predict().ravel()  # 已过滤的检测框
            label2 = int(detections[idx2[0]][4])
            conf2 = trackers[idx2[1]].confidence  # 获取另一个检测框的卡尔曼置信度

            # 计算两个检测框的 IOU
            iou_value = iou(bbox1, bbox2)
            if iou_value > iou_threshold:
                # if label1 == label2:
                #     # 标签相同，比较权重和置信度
                #     yolo_weight = 4
                #     kalman_weight = 6
                #
                #     # 计算两个检测框的可信度加权得分
                #     score1 = yolo_weight * conf1
                #     score2 = kalman_weight * conf2
                #     if score1 > score2:
                #         keep = True  # 保留当前 bbox1
                #         filtered_indices = [fi for fi in filtered_indices if not np.array_equal(fi, idx2)]  # 移除较小的 bbox2
                #     else:
                #         keep = False  # 另一框置信度更高，不保留 bbox1
                #         break
                # else:
                    # 标签不同，统计当前图像中两个标签的数量
                    count_label1 = label_counts.get(label1, 0)
                    count_label2 = label_counts.get(label2, 0)

                    # 比较标签的数量，保留数量较多的标签
                    if count_label1 < count_label2:
                        keep = False  # 标签1较少的被移除
                        break
                    elif count_label1 > count_label2:
                        filtered_indices = [fi for fi in filtered_indices if
                                            not np.array_equal(fi, idx2)]  # 移除数量较少的 bbox2
                    else:
                        # 标签相同，比较权重和置信度
                        yolo_weight = 5
                        kalman_weight = 5

                        # 计算两个检测框的可信度加权得分
                        score1 = yolo_weight * conf1
                        score2 = kalman_weight * conf2

                        if score1 > score2:
                            keep = True  # 保留置信度大的
                            filtered_indices = [fi for fi in filtered_indices if
                                                not np.array_equal(fi, idx2)]  # 移除之前的 bbox2
                        else:
                            keep = False  # 保留之前的 bbox
                            break

        # 如果检测框没有被删除，则保留
        if keep:
            filtered_indices.append(idx1)

    return filtered_indices
#
# 处理每个子文件夹
for heart_folder in glob.glob(os.path.join(valid_dir, "*")):
    images_dir = os.path.join(heart_folder, "images")
    labels_dir = os.path.join(heart_folder, "labels")  # 添加labels文件夹的路径
    if os.path.isdir(images_dir):
        # 重置跟踪器列表
        trackers = []

        # 定义输出子目录
        img_output_dir = os.path.join(output_base_dir, "img")
        lab_output_dir = os.path.join(output_base_dir, "lab")
        dimension_output_dir = os.path.join(output_base_dir, "dimension")
        yolo_lab_output_dir = os.path.join(output_base_dir, "yolo-lab")
        labels_output_dir = os.path.join(output_base_dir, "labels")  # 定义labels输出目录
        dataimage_output_dir = os.path.join(output_base_dir, "dataimage")  # 定义图输出目录

        # 创建输出目录
        for output_dir in [img_output_dir, lab_output_dir, dimension_output_dir, yolo_lab_output_dir,labels_output_dir,dataimage_output_dir]:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # 定义需要预测的文件夹
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        # 复制labels文件夹中的所有文件到输出目录
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            shutil.copy(label_file, labels_output_dir)  # 使用shutil.copy复制文件

        # 处理每帧图像
        for image_path in image_files:
            original_img = cv2.imread(image_path)
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            results = model(image_path)

            detections = []
            for result in results:
                for box in result.boxes:
                    det = box.xyxy[0].cpu().numpy()
                    label = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    detections.append([det[0], det[1], det[2], det[3], label, conf])

            # 使用匈牙利算法进行检测与跟踪的匹配
            current_detections = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
            matched_indices, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(current_detections, trackers)

            # 更新卡尔曼滤波器
            for m in matched_indices:
                trackers[m[1]].update(current_detections[m[0]])

            # 添加未匹配的检测为新的跟踪器
            for i in unmatched_detections:
                tracker = KalmanBoxTracker(current_detections[i])
                trackers.append(tracker)

            # 在这里进行第二次匹配（YOLO的结果与卡尔曼跟踪器的结果进行匹配）
            final_matched_indices, _, _ = associate_detections_to_trackers(current_detections, trackers)

            # 过滤位置相近的同类标签，删除面积较小的检测框
            filtered_matched_indices = filter_nearby_boxes(final_matched_indices, detections, trackers)

            # 保存YOLO格式的文本 (使用卡尔曼滤波器的预测框)
            base_filename = os.path.basename(image_path)
            with open(os.path.join(lab_output_dir, base_filename.replace('.jpg', '.txt')), 'w') as f:
                for m in filtered_matched_indices:
                    tracker_idx = m[1]
                    detection_idx = m[0]

                    tracker = trackers[tracker_idx]
                    bbox = tracker.predict().ravel()
                    label = int(detections[detection_idx][4])

                    x_center = (bbox[0] + bbox[2]) / 2 / width
                    y_center = (bbox[1] + bbox[3]) / 2 / height
                    w = (bbox[2] - bbox[0]) / width
                    h = (bbox[3] - bbox[1]) / height

                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # 保存 YOLO 识别的原始框
            with open(os.path.join(yolo_lab_output_dir, base_filename.replace('.jpg', '.txt')), 'w') as f:
                for det in detections:
                    bbox = det[:4]
                    label = int(det[4])

                    x_center = (bbox[0] + bbox[2]) / 2 / width
                    y_center = (bbox[1] + bbox[3]) / 2 / height
                    w = (bbox[2] - bbox[0]) / width
                    h = (bbox[3] - bbox[1]) / height

                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # 绘制检测框和标签
            for m in filtered_matched_indices:
                tracker_idx = m[1]
                detection_idx = m[0]

                tracker = trackers[tracker_idx]
                bbox = tracker.predict().ravel()  # 使用卡尔曼滤波器预测的边界框
                label = int(detections[detection_idx][4])  # 获取对应的标签
                color = label_colors.get(label, (255, 255, 255))  # 如果没有定义颜色，默认使用白色

                # 绘制边界框
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                # 绘制标签名称
                label_name = label_map.get(label, "Unknown")
                cv2.putText(img, label_name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 保存原图和带标签的图像
            cv2.imwrite(os.path.join(img_output_dir, base_filename), original_img)
            cv2.imwrite(os.path.join(dimension_output_dir, base_filename), img)

# 处理图像并使用YOLO和卡尔曼滤波进行跟踪
yolo_kh_lab_output_dir = os.path.join(output_base_dir, "kh-yolo-lab")
# 遍历每个 heart 文件夹
for heart_folder in os.listdir(valid_dir):
    heart_path = os.path.join(valid_dir, heart_folder)
    if not os.path.isdir(heart_path):
        continue
    image_dir = os.path.join(heart_path, "images")
    label_dir = os.path.join(heart_path, "labels")
    for image_path in os.listdir(image_dir):
        full_image_path = os.path.join(image_dir, image_path)
        original_img = cv2.imread(full_image_path)
        img = cv2.imread(full_image_path)
        height, width, _ = img.shape
        print(f"Processing: {full_image_path}")
        results = model(full_image_path)  # 调用YOLO模型进行检测

        detections = []
        for result in results:
            for box in result.boxes:
                det = box.xyxy[0].cpu().numpy()
                label = int(box.cls[0].item())
                conf = box.conf[0].item()
                detections.append([det[0], det[1], det[2], det[3], label, conf])

        # if len(detections) == 0:
        #     print("No detections found.")
        #     continue  # 如果没有检测结果，跳过该图像

        # 使用封装的卡尔曼滤波器跟踪逻辑
        final_detections, trackers, matched_indices = run_kalman_tracking(detections)

        # 保存 KHT-YOLO 格式的文本 (经过卡尔曼滤波和匈牙利算法的匹配结果)
        base_filename = os.path.basename(image_path)
        output_file_path = os.path.join(yolo_kh_lab_output_dir, base_filename.replace('.jpg', '.txt'))
        directory = os.path.dirname(output_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(output_file_path, 'w') as f:
            for m in matched_indices:
                tracker_idx = m[1]
                detection_idx = m[0]

                tracker = trackers[tracker_idx]
                bbox = tracker.predict().ravel()
                label = int(detections[detection_idx][4])  # 获取检测框的标签

                # 计算 YOLO 坐标 (相对于图片宽高归一化)
                x_center = (bbox[0] + bbox[2]) / 2 / width
                y_center = (bbox[1] + bbox[3]) / 2 / height
                w = (bbox[2] - bbox[0]) / width
                h = (bbox[3] - bbox[1]) / height

                f.write(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")



# # 在处理完当前图像后调用生成散点图的函数
# img_output_dir = os.path.join(output_base_dir, "img")
# lab_output_dir = './data/iou/preimages-yolo-240-iou-n1/lab'
# yolo_lab_output_dir = './data/iou/preimages-yolo-240-iou-n1/yolo-lab'
# labels_output_dir = './data/iou/preimages-yolo-240-iou-n1/labels'
# yolo_kh_lab_output_dir = './data/iou/preimages-yolo-240-iou-n1/kh-yolo-lab'
# dataimage_output_dir = './data/iou/preimages-yolo-240-iou-n1/dataimage'
# KHT-YOLO对比真实散点图
current_time = datetime.now().strftime('%Y%m%d%H%M')
Com_KHT_output_name = f"KHT-YOLO_PERSION_{current_time}.png"
generate_metric_scatter_plots(lab_output_dir, labels_output_dir, Com_KHT_output_name, dataimage_output_dir)
 # yolo对比真实散点图
current_time = datetime.now().strftime('%Y%m%d%H%M')
Com_YLv8_output_name = f"YOLOv8_PERSION_{current_time}.png"
generate_metric_scatter_plots(yolo_lab_output_dir, labels_output_dir, Com_YLv8_output_name, dataimage_output_dir)
 # yolo+k+h对比真实散点图
current_time = datetime.now().strftime('%Y%m%d%H%M')
Com_YLv8_output_name = f"YOLOv8+K+H_PERSION_{current_time}.png"
generate_metric_scatter_plots(yolo_kh_lab_output_dir, labels_output_dir, Com_YLv8_output_name, dataimage_output_dir)
 # 假设的路径
 # yolo对比折线图
current_time = datetime.now().strftime('%Y%m%d%H%M')
Eva_output_name = f"YOLOv8_KHT_{current_time}.png"
generate_evaluation_plots(lab_output_dir, yolo_lab_output_dir, yolo_kh_lab_output_dir, labels_output_dir, dataimage_output_dir, Eva_output_name)