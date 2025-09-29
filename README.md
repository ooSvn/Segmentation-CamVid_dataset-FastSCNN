# Urban Scene Segmentation using Fast-SCNN (CamVid)

This project implements a lightweight semantic segmentation system for urban scenes using the Fast-SCNN architecture, completed as part of a Neural Networks and Deep Learning course assignment. The goal is to segment 12 CamVid classes in real time and analyze performance with standard segmentation metrics, while conceptually comparing Fast-SCNN to encoder–decoder models like U-Net.

## 📌 Project Description

The aim of this project is to perform semantic segmentation on the CamVid dataset using a design tailored for real-time inference:

- Fast-SCNN: a two-path network with a high-resolution spatial path and a low-resolution context path, fused late for efficiency.
- Efficiency-first: extensive use of depthwise separable convolutions and MobileNetV2-style inverted residual blocks, with an overall downsampling factor of ×8 and an approximate parameter count of ~1.2M.
- Evaluation: metrics include mIoU, Dice, and Pixel Accuracy; qualitative visualizations are provided on validation samples.
- Training setup: 20 epochs, batch size 16, learning rate 1e-3, input 360×480; no data augmentation was used.
- Conceptual comparison with U-Net: highlights design and trade-offs (speed vs. accuracy).

## ⚙️ Tasks Overview

### 1) Data Preprocessing
- Dataset: CamVid (701 images with color-encoded masks).
- Classes (12 total): Bicyclist, Building, Car, Fence, Pedestrian, Pole, Road, Sidewalk, SignSymbol, Sky, Tree, Void.
- Converted color-encoded masks to class indices via an RGB-to-class mapping.
- Resized images and masks to 360×480 and normalized inputs with ImageNet mean/std.
- Data augmentation: not used (optional).
- Split: 90% training (630 images), 10% validation (71 images) with a fixed seed.

### 2) Metrics, Optimizer, and Loss
- Metrics: Pixel Accuracy, mean Intersection-over-Union (mIoU), and mean Dice Coefficient.
- Loss: Cross-Entropy.
- Optimizer: Adam (learning rate 1e-3).
- Learning rate schedule: step decay at epoch 20 (gamma = 0.5).

### 3) Model Architecture
- Learning to Downsample: initial 3×3 stride-2 conv followed by two depthwise separable convolutions with stride 2 (overall ×8 downsampling).
- Global Feature Extractor: stacks of inverted residual blocks (expansion 6; repeats 3–3–3) plus a Pyramid Pooling Module for multi-scale context.
- Feature Fusion Module: upsamples the context path and fuses it with the spatial path.
- Classifier: lightweight depthwise separable layers and a final projection, followed by upsampling to input resolution.
- Design goal: real-time performance with a compact model (~1.2M parameters).

### 4) Training & Evaluation
- Configuration: input 360×480, batch size 16, 20 epochs, Adam at 1e-3, step LR schedule, no augmentation.
- Monitoring: loss, Pixel Accuracy, mIoU, and Dice tracked for both training and validation.
- Visualization: qualitative comparison of predictions vs. ground truth on 10 validation images.

### 5) Results Analysis
- Validation performance (final epoch):
  - Loss: 0.3618
  - Pixel Accuracy: 0.8825
  - mIoU: 0.5760
  - Dice: 0.6957
- Per-class IoU (validation):
  - Bicyclist: 0.5456
  - Building: 0.7784
  - Car: 0.7218
  - Fence: 0.4395
  - Pedestrian: 0.2988
  - Pole: 0.1449
  - Road: 0.9383
  - Sidewalk: 0.7501
  - SignSymbol: 0.3102
  - Sky: 0.9196
  - Tree: 0.7226
  - Void: 0.3415
- Observations:
  - Strong performance on large, contiguous classes (Road, Sky, Building).
  - Lower IoU on thin/small objects (Pole, Sign, Pedestrian), which typically benefit from higher effective resolution, stronger multi-scale cues, and class-balanced losses.

### 6) Discussion & Insights
- Fast-SCNN vs. U-Net:
  - Architecture: two-path design with a single fusion stage vs. symmetric encoder–decoder with multi-scale skip connections.
  - Compute: Fast-SCNN is substantially lighter and designed for real-time inference; U-Net is often heavier but can achieve higher accuracy in certain domains.
  - Trade-off: Fast-SCNN offers an attractive balance for urban scenes where latency matters.
- Future improvements:
  - Class-balanced or focal losses to address small/thin classes.
  - Data augmentation (scales/crops, flips, color jitter) and longer training.
  - Alternative LR schedules (e.g., polynomial decay) and multi-scale training/inference.
  - Higher input resolution or pretraining on larger scene datasets.

## 📊 Example Results
- mIoU: 0.576, Dice: 0.696, Pixel Accuracy: 0.882 (validation).
- Best-performing classes: Road, Sky, Building (IoU ≥ 0.78; Road/Sky ≥ 0.91).
- Most challenging classes: Pole, Pedestrian, SignSymbol (IoU ≤ 0.31).
- Learning curves indicate steady improvement across epochs without augmentation.

## 🧠 Key Takeaways
- A two-path, lightweight design enables real-time semantic segmentation with competitive accuracy.
- Per-class IoU is crucial to understand strengths and weaknesses beyond overall accuracy.
- Large, contiguous structures are easier to segment; thin/rare objects need targeted strategies.
- For practical deployments, Fast-SCNN provides a strong speed–accuracy trade-off compared to heavier encoder–decoder models.
