import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 标签映射
label_map = {
    0: "Dummy",
    1: "D-head",
    2: "D-chest",
    3: "D-C-occlusion",
    4: "D-H-occlusion"
}


def read_yolo_labels(label_dir):
    labels = {}
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(label_file, 'r') as f:
            labels[os.path.basename(label_file)] = []
            for line in f:
                parts = list(map(float, line.split()))
                labels[os.path.basename(label_file)].append(parts)  # [class, x_center, y_center, width, height]
    return labels


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0


def evaluate(predictions, ground_truths, iou_threshold=0.4):
    TP = {label: 0 for label in label_map.keys()}
    FP = {label: 0 for label in label_map.keys()}
    FN = {label: 0 for label in label_map.keys()}
    IoUs = {label: [] for label in label_map.keys()}  # 用于记录每个类别的 IoU

    for image_name, preds in predictions.items():
        gt_boxes = ground_truths.get(image_name, [])
        matched_gt = set()

        for pred in preds:
            pred_box = [pred[1] - pred[3] / 2, pred[2] - pred[4] / 2, pred[1] + pred[3] / 2, pred[2] + pred[4] / 2]
            matched = False

            for i, gt in enumerate(gt_boxes):
                gt_box = [gt[1] - gt[3] / 2, gt[2] - gt[4] / 2, gt[1] + gt[3] / 2, gt[2] + gt[4] / 2]
                iou_value = calculate_iou(pred_box, gt_box)
                if iou_value >= iou_threshold and i not in matched_gt:
                    matched = True
                    matched_gt.add(i)
                    TP[int(pred[0])] += 1  # 更新对应类别的 TP
                    IoUs[int(pred[0])].append(iou_value)  # 记录 IoU
                    break

            if not matched:
                FP[int(pred[0])] += 1  # 更新对应类别的 FP

        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt:
                FN[int(gt[0])] += 1  # 更新对应类别的 FN

    precision = {label: TP[label] / (TP[label] + FP[label]) if (TP[label] + FP[label]) > 0 else 0 for label in
                 label_map.keys()}
    recall = {label: TP[label] / (TP[label] + FN[label]) if (TP[label] + FN[label]) > 0 else 0 for label in
              label_map.keys()}
    f1_score = {label: 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if
                (precision[label] + recall[label]) > 0 else 0 for label in label_map.keys()}
    iou_per_class = {label: np.mean(IoUs[label]) if IoUs[label] else 0 for label in label_map.keys()}  # 计算每个类别的平均 IoU

    return precision, recall, f1_score, iou_per_class


def generate_evaluation_plots(kalman_predictions_dir, yolo_predictions_dir, ykh_predictions_dir, ground_truths_dir,
                              output_dir, custom_filename):
    kalman_predictions = read_yolo_labels(kalman_predictions_dir)
    yolo_predictions = read_yolo_labels(yolo_predictions_dir)
    ykh_predictions = read_yolo_labels(ykh_predictions_dir)
    ground_truths = read_yolo_labels(ground_truths_dir)

    # 评估 Kalman (KHT-Y) 预测
    kalman_precision, kalman_recall, kalman_f1_score, kalman_iou_per_class = evaluate(kalman_predictions, ground_truths)

    # 评估 YOLOv8 (YLv8) 预测
    yolo_precision, yolo_recall, yolo_f1_score, yolo_iou_per_class = evaluate(yolo_predictions, ground_truths)

    # 评估 YKH 预测
    ykh_precision, ykh_recall, ykh_f1_score, ykh_iou_per_class = evaluate(ykh_predictions, ground_truths)

    # 打印评估结果
    print("KHT-Y (Kalman Filter) Results:")
    for label in label_map.keys():
        print(
            f"Class {label_map[label]} - Precision: {kalman_precision[label]:.4f}, Recall: {kalman_recall[label]:.4f}, F1 Score: {kalman_f1_score[label]:.4f}, Average IoU: {kalman_iou_per_class[label]:.4f}")

    print("\nYLv8 (YOLOv8) Results:")
    for label in label_map.keys():
        print(
            f"Class {label_map[label]} - Precision: {yolo_precision[label]:.4f}, Recall: {yolo_recall[label]:.4f}, F1 Score: {yolo_f1_score[label]:.4f}, Average IoU: {yolo_iou_per_class[label]:.4f}")

    print("\nYKH Results:")
    for label in label_map.keys():
        print(
            f"Class {label_map[label]} - Precision: {ykh_precision[label]:.4f}, Recall: {ykh_recall[label]:.4f}, F1 Score: {ykh_f1_score[label]:.4f}, Average IoU: {ykh_iou_per_class[label]:.4f}")

    # 生成子图
    labels = list(label_map.values())
    x = np.arange(len(labels))  # 标签索引

    # 生成图表数据
    kalman_precision_values = list(kalman_precision.values())
    kalman_recall_values = list(kalman_recall.values())
    kalman_f1_values = list(kalman_f1_score.values())
    kalman_iou_values = list(kalman_iou_per_class.values())

    yolo_precision_values = list(yolo_precision.values())
    yolo_recall_values = list(yolo_recall.values())
    yolo_f1_values = list(yolo_f1_score.values())
    yolo_iou_values = list(yolo_iou_per_class.values())

    ykh_precision_values = list(ykh_precision.values())
    ykh_recall_values = list(ykh_recall.values())
    ykh_f1_values = list(ykh_f1_score.values())
    ykh_iou_values = list(ykh_iou_per_class.values())

    # 设置子图布局
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # 2 行 2 列子图

    # 设置每条线的宽度
    line_width = 2

    # 调整后的颜色方案
    kalman_colors = {'precision': '#1f77b4', 'recall': '#ff7f0e', 'f1': '#2ca02c', 'iou': '#d62728'}
    yolo_colors = {'precision': '#9467bd', 'recall': '#8c564b', 'f1': '#e377c2', 'iou': '#7f7f7f'}
    ykh_colors = {'precision': '#bcbd22', 'recall': '#17becf', 'f1': '#d62728', 'iou': '#ff9896'}

    # 第一行第一个子图：Precision
    axs[0, 0].plot(x - 0.2, kalman_precision_values, marker='o', label='KHT-Y Precision', color=kalman_colors['precision'],
                   linewidth=line_width)
    axs[0, 0].plot(x, yolo_precision_values, marker='o', label='YLv8 Precision', color=yolo_colors['precision'],
                   linewidth=line_width)
    axs[0, 0].plot(x + 0.2, ykh_precision_values, marker='o', label='YKH Precision', color=ykh_colors['precision'],
                   linewidth=line_width)
    axs[0, 0].set_title('Precision by Class', fontsize=14, weight='bold')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, fontsize=12)
    axs[0, 0].legend(fontsize=10)

    # 第一行第二个子图：Recall
    axs[0, 1].plot(x - 0.2, kalman_recall_values, marker='o', label='KHT-Y Recall', color=kalman_colors['recall'],
                   linewidth=line_width)
    axs[0, 1].plot(x, yolo_recall_values, marker='o', label='YLv8 Recall', color=yolo_colors['recall'],
                   linewidth=line_width)
    axs[0, 1].plot(x + 0.2, ykh_recall_values, marker='o', label='YKH Recall', color=ykh_colors['recall'],
                   linewidth=line_width)
    axs[0, 1].set_title('Recall by Class', fontsize=14, weight='bold')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels, fontsize=12)
    axs[0, 1].legend(fontsize=10)

    # 第二行第一个子图：F1 Score
    axs[1, 0].plot(x - 0.2, kalman_f1_values, marker='o', label='KHT-Y F1 Score', color=kalman_colors['f1'],
                   linewidth=line_width)
    axs[1, 0].plot(x, yolo_f1_values, marker='o', label='YLv8 F1 Score', color=yolo_colors['f1'],
                   linewidth=line_width)
    axs[1, 0].plot(x + 0.2, ykh_f1_values, marker='o', label='YKH F1 Score', color=ykh_colors['f1'],
                   linewidth=line_width)
    axs[1, 0].set_title('F1 Score by Class', fontsize=14, weight='bold')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, fontsize=12)
    axs[1, 0].legend(fontsize=10)

    # 第二行第二个子图：IoU
    axs[1, 1].plot(x - 0.2, kalman_iou_values, marker='o', label='KHT-Y Average IoU', color=kalman_colors['iou'],
                   linewidth=line_width)
    axs[1, 1].plot(x, yolo_iou_values, marker='o', label='YLv8 Average IoU', color=yolo_colors['iou'],
                   linewidth=line_width)
    axs[1, 1].plot(x + 0.2, ykh_iou_values, marker='o', label='YKH Average IoU', color=ykh_colors['iou'],
                   linewidth=line_width)
    axs[1, 1].set_title('Average IoU by Class', fontsize=14, weight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, fontsize=12)
    axs[1, 1].legend(fontsize=10)

    # 调整子图布局以防止重叠
    plt.tight_layout()

    # 保存图片逻辑
    extension = '.png'
    counter = 0
    while True:
        output_path = os.path.join(output_dir, f'{custom_filename}{counter if counter > 0 else ""}{extension}')
        if not os.path.exists(output_path):
            plt.savefig(output_path)
            break
        counter += 1

    plt.show()

