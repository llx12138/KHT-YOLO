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

# 计算IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

# 计算 Precision、Recall 和 F1
def evaluate_single_file(preds, gts, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0
    IoUs = []
    matched_gt = set()

    # 遍历所有预测框
    for pred in preds:
        pred_box = [pred[1] - pred[3] / 2, pred[2] - pred[4] / 2, pred[1] + pred[3] / 2, pred[2] + pred[4] / 2]
        matched = False

        # 遍历所有真实框
        for i, gt in enumerate(gts):
            gt_box = [gt[1] - gt[3] / 2, gt[2] - gt[4] / 2, gt[1] + gt[3] / 2, gt[2] + gt[4] / 2]
            iou_value = calculate_iou(pred_box, gt_box)
            if iou_value >= iou_threshold and i not in matched_gt:
                matched = True
                matched_gt.add(i)
                TP += 1
                IoUs.append(iou_value)
                break

        if not matched:
            FP += 1

    FN = len(gts) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = np.mean(IoUs) if IoUs else 0

    return precision, recall, f1_score, average_iou
def generate_metric_scatter_plots(predictions_dir, ground_truths_dir, output_name, output_dir):
    predictions = read_yolo_labels(predictions_dir)
    ground_truths = read_yolo_labels(ground_truths_dir)

    # 存储每个标签的指标值
    metrics_per_label = {label: {'precision': [], 'recall': [], 'f1_score': [], 'iou_score': []} for label in
                         label_map.values()}

    # 遍历每个文件对，计算各指标
    for image_name in predictions.keys():
        preds = predictions.get(image_name, [])
        gts = ground_truths.get(image_name, [])

        # 记录每个标签的真实和预测数量
        gt_count = {label: 0 for label in label_map.values()}
        pred_count = {label: 0 for label in label_map.values()}

        # 计算真实标签的数量
        for gt in gts:
            label_name = label_map[int(gt[0])]
            gt_count[label_name] += 1

        # 根据标签分类计算指标
        for label_index, label_name in label_map.items():
            preds_for_label = [pred for pred in preds if int(pred[0]) == label_index]
            gts_for_label = [gt for gt in gts if int(gt[0]) == label_index]

            # 计算单个文件对该标签的指标
            precision, recall, f1_score, iou_score = evaluate_single_file(preds_for_label, gts_for_label)

            # 如果真实标签存在但预测标签缺失，则将预测值设为 0
            if gt_count[label_name] > 0 and len(preds_for_label) == 0:
                precision = 0
                recall = 0
                f1_score = 0
                iou_score = 0

            # 记录指标值
            metrics_per_label[label_name]['precision'].append(precision)
            metrics_per_label[label_name]['recall'].append(recall)
            metrics_per_label[label_name]['f1_score'].append(f1_score)
            metrics_per_label[label_name]['iou_score'].append(iou_score)

    # 过滤掉没有真实和预测标签的指标
    filtered_metrics = {label: metrics for label, metrics in metrics_per_label.items() if
                        any(np.array(metrics['precision']) > 0) or any(np.array(metrics['recall']) > 0)}

    # 横坐标为文件编号
    x = np.arange(1, len(predictions) + 1)

    # 根据标签数量动态创建子图
    n_labels = len(filtered_metrics)
    if n_labels == 0:
        print("No valid labels to plot. Exiting.")
        return  # 如果没有标签，直接退出或处理这个情况
    # 调整为更加合适的行列布局，例如 2 行 3 列
    n_cols = 3  # 列数
    n_rows = (n_labels + n_cols - 1) // n_cols  # 计算行数

    # 创建图表布局，行数和列数根据实际标签数进行设置
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # 确保 axes 是一个一维数组，即使是单行或单列也能正常访问
    axes = axes.ravel()

    # 用不同的颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']

    # 绘制每个标签的散点图
    for i, (label_name, metrics) in enumerate(filtered_metrics.items()):
        ax = axes[i]  # 动态选择子图

        for j, (metric_name, metric_values) in enumerate(metrics.items()):
            valid_y = np.array(metric_values)
            valid_x = x[valid_y > 0]  # 过滤掉 0 值的点
            valid_y = valid_y[valid_y > 0]  # 仅保留非零的 y 值

            if len(valid_x) > 0 and len(valid_y) > 0:  # 如果存在非零值再绘制
                ax.scatter(valid_x, valid_y, color=colors[j % len(colors)],
                           s=20, alpha=0.6, label=f'{label_name} - {metric_name}',
                           marker=markers[j % len(markers)])  # 设置透明度

        # 图表的细节设置
        ax.set_title(f'Metrics for {label_name}', fontsize=14, weight='bold')
        ax.set_xlabel('File Index', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True)
        ax.legend()

    # 隐藏掉多余的空子图
    for i in range(len(filtered_metrics), len(axes)):
        axes[i].axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存散点图
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.show()
