# CV+ML Paper Phylogenetic Taxonomy

4-level hierarchy: **Phylum > Class > Order > Genus**  
14 Phyla · ~110 Classes · ~380 Orders · Genus (~50% specific coverage)

---

## 1. Object Detection & Localization

- **2D Object Detection**
  - Region-based Detection (R-CNN, Faster R-CNN, Mask R-CNN)
  - One-stage Detection (YOLO, SSD, RetinaNet)
  - Anchor-free Detection (FCOS, CenterNet, CornerNet)
  - Transformer-based Detection (DETR, Deformable DETR, DAB-DETR)
  - Open-vocabulary Detection
  - Weakly/Semi-supervised Detection
  - Small Object Detection
- **3D Object Detection**
  - LiDAR-based 3D Detection
  - Camera-based 3D Detection (monocular, multi-view)
  - Multi-modal 3D Detection (LiDAR+Camera)
  - BEV (Bird's-Eye-View) Detection
- **Localization & Grounding**
  - Visual Grounding
  - Referring Expression Comprehension
  - Open-set / Zero-shot Detection

---

## 2. Segmentation

- **Image Segmentation**
  - Semantic Segmentation
  - Instance Segmentation
  - Panoptic Segmentation
  - Interactive Segmentation (SAM, click-based)
  - Open-vocabulary Segmentation
  - Weakly-supervised Segmentation
- **Video Segmentation**
  - Video Object Segmentation (VOS)
  - Video Instance Segmentation
  - Video Panoptic Segmentation
- **3D & Point Cloud Segmentation**
  - Point Cloud Semantic Segmentation
  - Point Cloud Instance Segmentation
  - 3D Scene Segmentation
- **Medical Image Segmentation**
  - Organ Segmentation
  - Lesion Segmentation
  - Cell/Tissue Segmentation

---

## 3. 3D Vision & Reconstruction

- **Depth & Stereo**
  - Monocular Depth Estimation
  - Stereo Matching & Depth
  - Multi-view Depth Estimation
  - Depth Completion
- **Multi-view Reconstruction**
  - Structure from Motion (SfM)
  - Multi-view Stereo (MVS)
  - Visual Localization (map-based)
  - Camera Pose Estimation
- **Neural Implicit Representations**
  - Neural Radiance Fields (NeRF)
  - Gaussian Splatting (3DGS)
  - Occupancy Networks
  - Signed Distance Functions
  - Dynamic / Deformable NeRF
- **Point Cloud Processing**
  - Point Cloud Classification
  - Point Cloud Registration (ICP variants)
  - Point Cloud Completion
  - Point Cloud Generation
- **3D Scene Understanding**
  - 3D Scene Reconstruction
  - Indoor Scene Understanding
  - Outdoor/LiDAR Mapping
  - 3D Shape Analysis & Generation

---

## 4. Image Recognition & Retrieval

- **Image Classification**
  - General Image Classification
  - Fine-grained Recognition
  - Multi-label Classification
  - Long-tail Recognition
  - Zero-shot / Open-set Recognition
- **Image Retrieval**
  - Hash-based Retrieval
  - Metric Learning
  - Content-based Retrieval (CBIR)
  - Visual Search & Re-ranking
- **Feature Matching & Correspondence**
  - Local Feature Detection & Description
  - Image Matching
  - Cross-domain Matching
- **Scene & Place Recognition**
  - Visual Place Recognition
  - Scene Classification
  - Landmark Recognition

---

## 5. Video & Motion Understanding

- **Action Recognition**
  - Trimmed Action Recognition
  - Skeleton-based Action Recognition
  - First-person / Egocentric Action
  - Multi-label Action Recognition
- **Temporal Action Analysis**
  - Temporal Action Detection
  - Action Segmentation
  - Action Localization
  - Activity / Event Detection
- **Motion Estimation**
  - Optical Flow Estimation
  - Scene Flow Estimation
  - Video Frame Interpolation
  - Motion Segmentation
- **Object Tracking**
  - Single Object Tracking (SOT)
  - Multi-Object Tracking (MOT)
  - 3D Multi-Object Tracking
  - Video Object Tracking with Language
- **Video Analysis**
  - Video Classification
  - Video Summarization
  - Video Anomaly Detection
  - Video Question Answering

---

## 6. Generative Models & Synthesis

- **Generative Adversarial Networks**
  - Unconditional Image Generation
  - Conditional Image Generation
  - Image-to-Image Translation
  - Style Transfer & Stylization
  - Video Generation (GAN)
  - 3D-aware Generation
- **Diffusion Models**
  - Unconditional Image Generation (Diffusion)
  - Text-to-Image Generation (Diffusion)
  - Image Editing with Diffusion
  - Video Generation (Diffusion)
  - 3D Generation (Diffusion)
  - Audio/Multi-modal Diffusion
- **VAE & Flow-based Models**
  - Variational Autoencoders
  - Normalizing Flows
  - Score-based Models
- **Image Editing & Manipulation**
  - Image Inpainting & Outpainting
  - Semantic Image Editing
  - Image Composition
  - Text-guided Image Manipulation
- **Face & Human Synthesis**
  - Face Generation & Synthesis
  - Face Reenactment & Swap
  - Human Body Generation
  - Talking Head Generation

---

## 7. Representation Learning

- **Self-supervised Learning**
  - Contrastive Learning (SimCLR, MoCo)
  - Masked Image Modeling (MAE, BEiT)
  - Self-supervised Pretraining
  - Multi-view / Multi-crop SSL
  - Predictive Self-supervised Learning
- **Transfer Learning**
  - Domain Adaptation (UDA, SFDA)
  - Domain Generalization
  - Fine-tuning Methods (adapter, LoRA)
  - Prompt Learning / Prompt Tuning
- **Foundation Visual Models**
  - Vision Transformer Pretraining
  - Visual Foundation Models (SAM, DINOv2)
  - Universal Representations
- **Embedding & Similarity Learning**
  - Metric Learning (triplet, contrastive loss)
  - Embedding Space Analysis
  - Disentangled Representation

---

## 8. Vision-Language & Multimodal

- **Image-Text Understanding**
  - Image Captioning
  - Visual Question Answering (VQA)
  - Visual Reasoning & Entailment
  - Cross-modal Retrieval (image↔text)
- **Vision-Language Pretraining**
  - Contrastive Language-Image (CLIP)
  - Masked Vision-Language Modeling
  - Large Vision-Language Models (LLaVA, GPT-4V)
  - Visual Instruction Tuning
- **Text-to-Visual Generation**
  - Text-to-Image Generation
  - Text-guided Video Generation
  - Layout-conditioned Generation
- **Multimodal Fusion & Reasoning**
  - Multimodal Feature Fusion
  - Visual Commonsense Reasoning
  - Science / Chart Understanding
  - Document Visual QA

---

## 9. Low-level Vision

- **Image Restoration**
  - Super-resolution
  - Image Denoising (Gaussian, real-world)
  - Image Deblurring
  - Deraining & Dehazing
  - JPEG Artifact Removal
  - Blind Restoration
- **Image Enhancement**
  - Low-light Image Enhancement
  - HDR Imaging & Tone Mapping
  - Color Enhancement & Correction
  - Exposure Correction
- **Image Compression**
  - Learned Image Compression
  - Learned Video Compression
  - Compression Artifact Reduction
- **Computational Photography**
  - Image Stitching & Mosaicking
  - Computational Bokeh / Depth-of-field
  - Reflection / Flare Removal
  - Night Photography Enhancement

---

## 10. Human-centric Vision

- **Face Analysis**
  - Face Detection & Alignment
  - Face Recognition & Verification
  - Face Anti-spoofing & Deepfake Detection
  - Facial Attribute Analysis
  - Facial Landmark Detection
- **Human Pose & Body**
  - 2D Human Pose Estimation
  - 3D Human Pose Estimation
  - Human Body Parsing & Part Segmentation
  - Human Mesh Recovery (SMPL-based)
  - Hand Pose Estimation
- **Person Re-identification**
  - Pedestrian Re-identification
  - Vehicle Re-identification
  - Cloth-changing ReID
- **Gesture & Interaction**
  - Gesture Recognition
  - Sign Language Recognition
  - Hand Tracking
- **Crowd & Activity**
  - Crowd Counting & Density Estimation
  - Group Activity Recognition
  - Human-Object Interaction (HOI)
- **Affective Computing**
  - Emotion / Facial Expression Recognition
  - Gaze Estimation
  - Age Estimation

---

## 11. Deep Learning Architecture

- **Convolutional Architectures**
  - Efficient CNN Design (MobileNet, EfficientNet)
  - Depthwise & Grouped Convolution
  - Multi-scale Feature Learning (FPN, PANet)
  - Skip Connections & Dense Connectivity
- **Transformer Architectures**
  - Vision Transformer (ViT, DeiT)
  - Swin Transformer & Hierarchical ViT
  - Hybrid CNN-Transformer
  - Efficient Attention Mechanisms
- **Graph & Relational Networks**
  - Graph Convolutional Networks (GCN)
  - Scene Graph Generation
  - Relational Reasoning Networks
- **Neural Architecture Search**
  - Gradient-based NAS (DARTS)
  - Evolutionary / RL-based NAS
  - Once-for-all & Hardware-aware NAS
- **Attention & Memory**
  - Self-attention Mechanisms (CBAM, SE)
  - Cross-attention & Co-attention
  - External Memory Networks
- **MLP & Alternative Architectures**
  - MLP-Mixer & Pure MLP Architectures
  - State Space Models (Mamba)
  - Capsule Networks

---

## 12. Training & Learning Methods

- **Optimization**
  - Gradient Descent & Adaptive Optimizers
  - Loss Function Design
  - Second-order Optimization
  - Learning Rate Scheduling
- **Data Augmentation**
  - Classical Augmentation (Flip, Crop, Color)
  - Mixup / CutMix / Mosaic
  - AutoAugment & RandAugment
  - Synthetic Data Generation
- **Supervised Learning Strategies**
  - Label Smoothing & Regularization
  - Long-tail / Class-imbalanced Learning
  - Noisy Label Learning
  - Curriculum Learning
- **Semi-supervised & Weakly-supervised**
  - Semi-supervised Learning (pseudo-label)
  - Weakly-supervised Learning
  - Active Learning
  - Label-efficient Learning
- **Few-shot & Meta-learning**
  - Few-shot Classification
  - Meta-learning (MAML, Prototypical)
  - In-context Learning
  - Prompt-based Few-shot
- **Knowledge Distillation**
  - Offline Knowledge Distillation
  - Online / Mutual Distillation
  - Feature-based Distillation
- **Continual & Lifelong Learning**
  - Catastrophic Forgetting Mitigation
  - Class-incremental Learning
  - Task-incremental Learning
  - Replay-based Methods

---

## 13. Efficient & Robust ML

- **Model Compression**
  - Network Pruning (structured, unstructured)
  - Network Quantization (PTQ, QAT)
  - Knowledge Distillation for Compression
  - Low-rank Approximation
- **Efficient Inference**
  - Hardware-aware Optimization
  - TensorRT & Deployment
  - Early Exit & Dynamic Inference
  - Token Pruning for Transformers
- **Adversarial Robustness**
  - Adversarial Attack Methods
  - Adversarial Defense & Training
  - Certified Robustness
  - Backdoor & Trojan Attacks
- **Reliability & Trustworthy AI**
  - Out-of-distribution (OOD) Detection
  - Uncertainty Estimation & Calibration
  - Anomaly Detection
  - Fairness & Bias Mitigation
  - Explainability & Interpretability (Grad-CAM, LIME)
- **Privacy & Federated Learning**
  - Federated Learning
  - Differential Privacy
  - Data-free / Data-free Distillation
  - Membership Inference Defense

---

## 14. Application Domains

- **Medical & Clinical Imaging**
  - Medical Image Segmentation
  - Medical Image Classification / Diagnosis
  - Medical Image Registration
  - Radiology (X-ray, CT, MRI)
  - Pathology & Histology
  - Ophthalmology (Fundus, OCT)
  - Surgical & Endoscopic Vision
- **Autonomous Driving**
  - Autonomous Driving Perception
  - 3D Detection for Driving (LiDAR/Camera)
  - Lane Detection & Segmentation
  - BEV Perception & HD Map
  - End-to-end Autonomous Driving
  - Traffic Analysis
- **Remote Sensing & Geospatial**
  - Satellite Image Analysis
  - Aerial Image Analysis
  - Change Detection
  - Hyperspectral / SAR Analysis
  - Earth Observation
- **Document & Scene Text**
  - Scene Text Detection
  - Scene Text Recognition (OCR)
  - Document Layout Analysis
  - Visual Document Understanding
  - Handwriting Recognition
- **Other Application Domains**
  - Agricultural & Environmental Vision
  - Fashion & Retail Vision
  - Industrial Inspection & Defect Detection
  - Scientific Imaging (Astronomy, Biology)
  - AR/VR & Mixed Reality

---

## 0. Other / Unclassified *(catchall)*

---

## 분류 우선순위 (specificity ordering)

```
1. Editorial / Proceedings (즉시 처리)
2. Medical Imaging (강한 도메인 식별자)
3. Autonomous Driving (강한 도메인 식별자)
4. Remote Sensing (강한 도메인 식별자)
5. NeRF / Gaussian Splatting (극도로 구체적)
6. Diffusion Models (강한 식별자)
7. GAN / Image Synthesis
8. Segmentation (강한 task 식별자)
9. 3D Object Detection
10. 2D Object Detection
11. Human-centric (face, pose, ReID)
12. Vision-Language / Multimodal
13. Video & Motion (action, tracking, flow)
14. Low-level Vision
15. Self-supervised / Representation Learning
16. Image Recognition & Retrieval
17. Deep Learning Architecture
18. Training Methods
19. Efficient & Robust ML
20. Fallback → Other/Unclassified
```
