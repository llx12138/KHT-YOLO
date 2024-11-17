## Methodology

### A. KHT-YOLO Framework

The semi-supervised framework for key frame sequence annotation of CPR videos (KHT-YOLO), as shown in Figure 2, extracts key frames from CPR assessment videos, manually annotates some of the frames, and divides the labeled dataset into training and validation sets. YOLO11n is used for recognition. The Kalman filter and Hungarian matching algorithm are employed for semi-supervised automatic annotation of the unlabeled frames.

When the recognition results are inconsistent or confusing, Intersection over Union (IoU) edge matching is used to determine whether the positions are similar. If they are similar and the recognition results are consistent, the smallest target identified by KHT is retained. If they are not consistent, the auxiliary view is examined, and its weighting is set to 10 to correct the inconsistent recognition. If other angles cannot provide clarity, the confidence scores of KHT (weighted at 5) and YOLO11n (weighted at 5) are compared, and the result with the highest confidence score is selected as the final recognition result.

**Algorithm 1 KHT-YOLO Framework**
1. **Input:** CPR training assessment video
2. **Output:** Key frame sequence annotation
3. **Step 1:** Preprocess Video
   - Convert the video format and adjust the resolution to prepare it for model processing.
4. **Step 2:** Annotate Data
   - Manually label key actions and feature points in selected frames for training the model.
5. **Step 3:** Train YOLO11n Model
   - Apply data augmentations (e.g., blurring, grayscale conversion, brightness adjustment) to enhance generalization capabilities.
6. **Step 4:** YOLO11n Detection
   - Use the YOLO11n model to detect key actions and feature points in the video frames.
7. **Step 5:** KHT Matching
   - Predict key points of the next frame using KHT matching.
   - Validate matches using Intersection over Union (IoU) and iteratively update predictions.

## Results

From the scatter plots, it can be observed that YOLO11n, YOLO11n combined with the original Kalman filter and Hungarian matching algorithm, and the proposed KHT-YOLO (KHT-Y) exhibit a highly dense distribution within the presented metrics and data ranges. The data points are primarily concentrated in the higher value ranges, indicating that these models perform well across multiple metrics, with particularly stable performance in terms of accuracy. Furthermore, the consistency among the metrics further demonstrates the robustness and overall excellent performance of the models under various conditions.

## Conclusion

This study presents a semi-supervised framework for temporal key frame annotation in CPR videos, named KHT-YOLO, which is based on OpenCV, YOLO11, an optimized Kalman filter, and the Hungarian algorithm. Designed for semi-supervised object detection and annotation in cardiopulmonary resuscitation (CPR) assessment videos, this is the first application of YOLO11 in CPR assessment, enabling large-scale key frame object detection with minimal manual annotation. Experimental results indicate that KHT-YOLO outperforms YOLO11 (YL11n) and its Kalman-filter-combined variant (YKH) across precision, F1 score, recall, IoU, and mAP50 metrics. In the Dummy task, KHT-YOLO achieved a precision of 0.999, with an average precision of 0.973, representing improvements of 2.2% and 2.7% over YL11n and YKH, respectively. Notably, KHT-YOLO demonstrated superior robustness in complex occlusion scenarios, such as in the D-H-occlusion task, where it attained an IoU of 0.748.

In the future, this research will expand the CPR training dataset using this method and incorporate a detection module to monitor the frequency and depth of chest compressions and artificial respirations while assessing students’ performance.

## Project Directory Structure

The project is organized into several directories to facilitate development, testing, and deployment of the CPR detection system using the KHT-YOLO methodology. Below is a detailed explanation of each directory and its contents:

### `code`
This directory contains the main codebase of the project. It includes scripts and modules essential for running the CPR detection system.

- **`CPRDetection`**: 
  - **Core Implementation of KHT-YOLO**
  - **Files:**
    - `Compare_s.py`: Script for comparing different aspects or results.
    - `EvaluatingIndicator1.py`: File for evaluating specific indicators or metrics relevant to the CPR detection system.
    - `K_H.py`: Implements the Kalman filter (`K`) and Hungarian algorithm (`H`).
    - `YOLO+IOU.py`: Main function or script implementing the YOLO model enhanced with IOU calculations, possibly version 7.
  - **Subdirectories:**
    - `docker`: Docker-related configurations or scripts for containerized deployment.
    - `docs`: Documentation files, including guides, manuals, or API documentation.
    - `examples`: Example usage cases or sample inputs/outputs demonstrating how the system works.
    - `tests`: Test cases and scripts to ensure the functionality and reliability of the codebase.
    - `ultralytics`: Additional tools or libraries related to deep learning frameworks like Ultralytics' YOLO11.

### `data`
This directory stores the processed data and results generated during the execution of the program.

- **Subdirectories:**
  - `iou/preimages-yolo-240-i...`: Contains preprocessed images ready for input into the YOLO model, with dimensions adjusted to 240 pixels.
  - `KHT-YOLO_PERSION_2024110...`: Results or datasets associated with the KHT-YOLO model.
  - `YOLO11n+K+H_PERSION_202...`: Results or datasets associated with the YOLO11n model combined with the Kalman filter and Hungarian algorithm.
  - `YOLO11n_KHT_202411051659...`: Results or datasets associated with the YOLO11n model enhanced with KHT.
  - `YOLO11n_PERSION_20241105...`: Results or datasets associated with the YOLO11n model.

### `datasets`
This directory contains the raw and processed datasets used for training and validating the CPR detection system.

- **Subdirectories:**
  - `raw`: Raw data collected from CPR training videos.
  - `processed`: Processed data ready for model training and validation.
  - `annotations`: Manually annotated key frames and feature points.

### `LICENSE`
Legal document outlining the licensing terms under which the software is distributed.

### `CITATION.cff`
Citation file providing information about how to cite the project if used in academic work.

### `README.md`
A markdown file containing an overview of the project, installation instructions, and usage examples.

---

This structure ensures a clear and organized approach to developing, testing, and deploying the CPR detection system, making it easier for developers and researchers to navigate and contribute to the project.
