"""
CVML Genus Rules
=================
Provides sub-Order (Genus) classification for the top 30 Orders.
Called AFTER classify() assigns (Phylum, Class, Order).

Usage:
    from classify import classify
    from genus_rules import assign_genus

    phylum, class_, order = classify(title)
    genus = assign_genus(phylum, class_, order, title)
"""

from __future__ import annotations


def _t(title: str) -> str:
    return ' ' + title.lower() + ' '


def _any(t: str, kws: list[str]) -> bool:
    return any(kw in t for kw in kws)


# ---------------------------------------------------------------------------
# Genus assignment
# ---------------------------------------------------------------------------

def assign_genus(phylum: str, class_: str, order: str, title: str) -> str:
    """
    Return a Genus string for the given (Phylum, Class, Order, title).
    Returns '(general)' when no specific Genus applies.

    Orders covered:
      1.  Monocular Depth Estimation
      2.  Neural Radiance Fields
      3.  Gaussian Splatting
      4.  Image Diffusion / Video Diffusion / 3D Diffusion / Audio Diffusion / Medical Diffusion
      5.  Image Classification
      6.  Video Action Recognition
      7.  Semantic Segmentation
      8.  Instance Segmentation
      9.  Generic Object Detection
     10.  Transformer-based Detection
     11.  Person Re-Identification
     12.  2D Human Pose Estimation
     13.  3D Human Pose & Shape Estimation
     14.  Face Recognition & Verification
     15.  Optical Flow Estimation
     16.  Multi-Object Tracking
     17.  Single Object Tracking
     18.  Contrastive Self-supervised Learning  (now also covers Video SSL Genus)
     19.  Masked Autoencoders (MAE)
     20.  Domain Adaptation
     21.  Knowledge Distillation
     22.  Image Restoration
     23.  Single Image Super-Resolution
     24.  Vision Transformers (ViT)              (now also covers Efficient ViT Genus)
     25.  Hierarchical Vision Transformers
     26.  Few-shot Learning
     27.  Adversarial Attacks
     28.  Medical Image Segmentation
     29.  Point Cloud Classification
     30.  Point Cloud Object Detection
     31.  Graph Neural Networks                  (now also covers Spatial-Temporal GNN Genus)
     32.  General 3D Vision                      (renamed from "3D Scene Understanding" Order)
     33.  3D Shape Analysis
     34.  AR/VR Scene Understanding
     35.  General Image Segmentation             (renamed from "Image Segmentation" Order)
     36.  General VQA                            (renamed from "Visual Question Answering" Order)
    """
    t = _t(title)

    # ------------------------------------------------------------------
    # 1. Monocular Depth Estimation (also covers Self-supervised variant)
    # ------------------------------------------------------------------
    if order in ('Monocular Depth Estimation', 'Self-supervised Depth Estimation',
                 'Video Depth Estimation', 'Foundation Model Depth', 'Depth Completion'):
        if _any(t, ['self-supervised', 'self supervised', 'unsupervised']):
            return 'Self-supervised Depth'
        if _any(t, ['foundation', 'zero-shot', 'generalist', 'depth anything']):
            return 'Foundation Model Depth'
        if _any(t, ['affine-invariant', 'scale-invariant', 'metric depth', 'absolute depth']):
            return 'Metric Depth Estimation'
        if _any(t, ['indoor', 'nyu', 'nyud']):
            return 'Indoor Depth Estimation'
        if _any(t, ['outdoor', 'kitti', 'driving']):
            return 'Outdoor Depth Estimation'
        if _any(t, ['uncertainty', 'probabilistic', 'bayesian']):
            return 'Probabilistic Depth Estimation'
        if _any(t, ['transformer', 'vit', 'attention']):
            return 'Transformer-based Depth'
        return '(general)'

    # ------------------------------------------------------------------
    # 2. Neural Radiance Fields
    #    Order is intentionally flat (always 'Neural Radiance Fields');
    #    every variant — Dynamic / Human / Editing / Large-scale /
    #    Efficient / etc. — lives here at the Genus level.
    # ------------------------------------------------------------------
    if order == 'Neural Radiance Fields':
        # Specific axes first so they don't collide with the broader buckets.
        if _any(t, ['dynamic', 'deformable', 'non-rigid', '4d nerf', 'time-varying']):
            return 'Dynamic NeRF'
        if _any(t, ['avatar', 'human nerf', 'human body', 'face nerf',
                    'head nerf', 'cloth', 'sifu', 'animatable']):
            return 'Human NeRF'
        if _any(t, ['edit', 'styliz', 'manipulat', 'inversion nerf']):
            return 'NeRF Editing'
        if _any(t, ['few-shot', 'sparse view', 'generalizable', 'generalized nerf', 'one-shot']):
            return 'Few-shot / Generalizable NeRF'
        if _any(t, ['self-supervised', 'unsupervised']):
            return 'Self-supervised NeRF'
        if _any(t, ['indoor', 'room', 'scene-level']):
            return 'Indoor Scene NeRF'
        if _any(t, ['outdoor', 'large-scale', 'city', 'urban', 'unbounded', 'block-nerf']):
            return 'Large-scale NeRF'
        if _any(t, ['instant', 'hash', 'fast', 'real-time', 'accelerat',
                    'efficien', 'compress', 'compact nerf']):
            return 'Efficient NeRF'
        if _any(t, ['semantic', 'language', 'clip']):
            return 'Semantic NeRF'
        return '(general)'

    # ------------------------------------------------------------------
    # 3. Gaussian Splatting (same flat-order policy as NeRF above)
    # ------------------------------------------------------------------
    if order == 'Gaussian Splatting':
        if _any(t, ['dynamic', '4d gaussian', 'deformable', 'time-varying']):
            return 'Dynamic Gaussian Splatting'
        if _any(t, ['avatar', 'human', 'body', 'cloth', 'animatable']):
            return 'Human Gaussian Splatting'
        if _any(t, ['edit', 'styliz', 'manipulat']):
            return 'Gaussian Splatting Editing'
        if _any(t, ['few-shot', 'sparse', 'generaliz', 'one-shot']):
            return 'Few-shot Gaussian Splatting'
        if _any(t, ['large-scale', 'city', 'urban', 'outdoor', 'unbounded']):
            return 'Large-scale Gaussian Splatting'
        if _any(t, ['efficient', 'compact', 'compress', 'lightweight']):
            return 'Compact Gaussian Splatting'
        if _any(t, ['semantic', 'language', 'clip', 'open-vocabulary']):
            return 'Semantic Gaussian Splatting'
        if _any(t, ['surface', 'mesh', 'geometry', 'normal']):
            return 'Geometry-aware Gaussian Splatting'
        return '(general)'

    # ------------------------------------------------------------------
    # 4. Diffusion Models
    #    Order is split by *modality* (Image / Video / 3D / Audio / Medical).
    #    The earlier Order "Diffusion Models" (which clashed with the Class
    #    name) has been renamed to "Image Diffusion". Method/conditioning
    #    variants — Text-to-Image, Image Editing, Latent Diffusion,
    #    Score / Flow Matching, etc. — live here at the Genus level.
    # ------------------------------------------------------------------
    if order == 'Image Diffusion':
        # Conditioning / method variants for image-modality diffusion.
        if _any(t, ['text-to-image', 'text to image', 'text-guided',
                    'text-driven', 'dall-e', 'imagen', 'parti ',
                    'prompt-to-image']):
            return 'Text-to-Image'
        if _any(t, ['edit', 'inpaint', 'outpaint', 'manipulat',
                    'instructpix2pix', 'instruct-pix2pix', 'drag-based']):
            return 'Image Editing (Diffusion)'
        if _any(t, ['latent diffusion', 'stable diffusion', 'ldm ',
                    'latent space diffusion']):
            return 'Latent Diffusion'
        if _any(t, ['score matching', 'score-based', 'flow matching',
                    'rectified flow', 'continuous flow',
                    'stochastic differential equation', 'sde-based',
                    'consistency model']):
            return 'Score / Flow Matching'
        if _any(t, ['controlnet', 'control-net', 'conditional diffusion',
                    'classifier-free guidance', 'cfg guidance', 'guidance']):
            return 'Conditional / Controllable Diffusion'
        if _any(t, ['fast sample', 'accelerat', 'ddim', 'distil',
                    'efficient diffusion', 'few-step', 'one-step']):
            return 'Fast Sampling / Distillation'
        if _any(t, ['unconditional', 'pure noise', 'noise-to-image']):
            return 'Unconditional'
        return '(general)'

    if order == 'Video Diffusion':
        if _any(t, ['text-to-video', 'text to video']):
            return 'Text-to-Video'
        if _any(t, ['edit', 'inpaint', 'outpaint', 'manipulat']):
            return 'Video Editing (Diffusion)'
        if _any(t, ['long video', 'long-form', 'temporal consistency']):
            return 'Long-form Video Diffusion'
        if _any(t, ['fast', 'accelerat', 'efficient', 'distil']):
            return 'Efficient Video Diffusion'
        if _any(t, ['controlnet', 'control', 'conditional']):
            return 'Conditional Video Diffusion'
        return '(general)'

    if order == '3D Diffusion':
        if _any(t, ['point cloud']):
            return 'Point Cloud Diffusion'
        if _any(t, ['shape', 'mesh']):
            return '3D Shape Diffusion'
        if _any(t, ['scene', 'room', 'indoor']):
            return '3D Scene Diffusion'
        if _any(t, ['mol', 'protein', 'drug']):
            return 'Molecular Diffusion'
        if _any(t, ['text-to-3d', 'text to 3d']):
            return 'Text-to-3D'
        return '(general)'

    if order == 'Audio Diffusion':
        if _any(t, ['speech']):
            return 'Speech Diffusion'
        if _any(t, ['music']):
            return 'Music Diffusion'
        if _any(t, ['text-to-audio', 'text to audio', 'text-to-music',
                    'text-to-speech']):
            return 'Text-to-Audio'
        return '(general)'

    if order == 'Medical Diffusion':
        if _any(t, [' mri ']):
            return 'MRI Diffusion'
        if _any(t, ['ct ', 'ct scan']):
            return 'CT Diffusion'
        if _any(t, ['pathology', 'histolog']):
            return 'Pathology Diffusion'
        if _any(t, ['segment']):
            return 'Diffusion-based Medical Segmentation'
        return '(general)'

    # ------------------------------------------------------------------
    # 5. Image Classification
    # ------------------------------------------------------------------
    if order == 'Image Classification':
        if _any(t, ['long-tail', 'imbalanced', 'class imbalance', 'tail']):
            return 'Long-tail Classification'
        if _any(t, ['robustness', 'out-of-distribution', 'distribution shift', 'corruption']):
            return 'Robust Classification'
        if _any(t, ['noisy label', 'label noise']):
            return 'Noisy-label Classification'
        if _any(t, ['efficient', 'mobile', 'lightweight', 'compact']):
            return 'Efficient Image Classifiers'
        if _any(t, ['transformer', 'vit', 'patch']):
            return 'Transformer-based Classification'
        if _any(t, ['multi-label', 'multiple label']):
            return 'Multi-label Classification'
        if _any(t, ['imagenet', 'large-scale', 'top-1', 'top-5']):
            return 'Large-scale Classification (ImageNet)'
        return '(general)'

    # ------------------------------------------------------------------
    # 6. Video Action Recognition
    # ------------------------------------------------------------------
    if order == 'Video Action Recognition':
        if _any(t, ['skeleton', 'graph', 'gcn', 'bone', 'joint']):
            return 'Skeleton-based Action Recognition'
        if _any(t, ['first-person', 'egocentric', 'ego', 'hand-object']):
            return 'Egocentric Action Recognition'
        if _any(t, ['two-stream', 'rgb flow', 'optical flow']):
            return 'Two-stream Action Recognition'
        if _any(t, ['transformer', 'vit', 'attention', 'video transformer']):
            return 'Transformer-based Action Recognition'
        if _any(t, ['few-shot', 'zero-shot', 'novel class']):
            return 'Few/Zero-shot Action Recognition'
        if _any(t, ['self-supervised', 'contrastive', 'pretrain']):
            return 'Self-supervised Action Recognition'
        if _any(t, ['sport', 'fitness', 'workout']):
            return 'Sports Action Recognition'
        return '(general)'

    # ------------------------------------------------------------------
    # 7. Semantic Segmentation
    # ------------------------------------------------------------------
    if order == 'Semantic Segmentation':
        if _any(t, ['domain adapt', 'synthetic', 'sim-to-real', 'synth']):
            return 'Domain Adaptive Segmentation'
        if _any(t, ['transformer', 'segformer', 'maskformer', 'mask2former', 'vit']):
            return 'Transformer-based Segmentation'
        if _any(t, ['open-vocabulary', 'open-set', 'zero-shot', 'language', 'clip']):
            return 'Open-vocabulary Segmentation'
        if _any(t, ['real-time', 'efficient', 'lightweight', 'fast']):
            return 'Real-time Semantic Segmentation'
        if _any(t, ['3d', 'point cloud', 'lidar']):
            return '3D Semantic Segmentation'
        if _any(t, ['weakly', 'semi-supervised', 'image-level']):
            return 'Weakly/Semi-supervised Segmentation'
        if _any(t, ['urban', 'cityscapes', 'street', 'driving']):
            return 'Urban Scene Parsing'
        return '(general)'

    # ------------------------------------------------------------------
    # 8. Instance Segmentation
    # ------------------------------------------------------------------
    if order == 'Instance Segmentation':
        if _any(t, ['transformer', 'mask2former', 'maskformer', 'solq', 'queryinst']):
            return 'Transformer-based Instance Segmentation'
        if _any(t, ['open-vocabulary', 'open-set', 'zero-shot', 'language']):
            return 'Open-vocabulary Instance Segmentation'
        if _any(t, ['real-time', 'efficient', 'yolo-seg', 'yolact']):
            return 'Real-time Instance Segmentation'
        if _any(t, ['few-shot', 'novel class']):
            return 'Few-shot Instance Segmentation'
        if _any(t, ['3d', 'point cloud', 'lidar', 'scene', 'indoor']):
            return '3D Instance Segmentation'
        return '(general)'

    # ------------------------------------------------------------------
    # 9. Generic Object Detection
    # ------------------------------------------------------------------
    if order == 'Generic Object Detection':
        if _any(t, ['real-time', 'fast', 'efficien', 'edge']):
            return 'Real-time Detection'
        if _any(t, ['aerial', 'drone', 'satellite', 'remote sens']):
            return 'Aerial/Remote-sensing Detection'
        if _any(t, ['few-shot', 'novel', 'zero-shot']):
            return 'Few/Zero-shot Detection'
        if _any(t, ['weakly', 'image-level', 'semi-supervised']):
            return 'Weakly/Semi-supervised Detection'
        if _any(t, ['thermal', 'infrared', 'rgb-d', 'depth-based']):
            return 'Multi-modality Detection'
        return '(general)'

    # ------------------------------------------------------------------
    # 10. Transformer-based Detection
    # ------------------------------------------------------------------
    if order == 'Transformer-based Detection':
        if _any(t, ['deformable detr', 'dino detect', 'dn-detr', 'conditional detr', 'anchor detr']):
            return 'Deformable DETR Family'
        if _any(t, ['open-vocabulary', 'grounding', 'language', 'clip']):
            return 'Open-vocabulary Transformer Detection'
        if _any(t, ['end-to-end', 'bipartite match', 'hungarian']):
            return 'End-to-end Transformer Detection'
        if _any(t, ['efficient', 'fast', 'lite detr', 'real-time detr']):
            return 'Efficient DETR'
        return '(general)'

    # ------------------------------------------------------------------
    # 11. Person Re-Identification
    # ------------------------------------------------------------------
    if order == 'Person Re-Identification':
        if _any(t, ['occluded', 'partial', 'incomplete']):
            return 'Occluded Person ReID'
        if _any(t, ['text', 'language', 'text-to-person', 'irra']):
            return 'Text-guided Person ReID'
        if _any(t, ['unsupervised', 'self-supervised', 'pseudo label']):
            return 'Unsupervised Person ReID'
        if _any(t, ['cloth-changing', 'long-term', 'cloth-consistent']):
            return 'Cloth-changing Long-term ReID'
        if _any(t, ['domain adapt', 'cross-domain', 'source-free']):
            return 'Domain Adaptive ReID'
        if _any(t, ['3d', 'spatial', 'lidar', 'point cloud']):
            return '3D-aware ReID'
        return '(general)'

    # ------------------------------------------------------------------
    # 12. 2D Human Pose Estimation
    # ------------------------------------------------------------------
    if order == '2D Human Pose Estimation':
        if _any(t, ['transformer', 'vit', 'attention', 'vitpose', 'tokenpose']):
            return 'Transformer-based Pose Estimation'
        if _any(t, ['bottom-up', 'associative embed', 'heatmap-free', 'openpose']):
            return 'Bottom-up Multi-person Pose'
        if _any(t, ['top-down', 'hrnet', 'simple baseline']):
            return 'Top-down Single-person Pose'
        if _any(t, ['animal', 'ap-10k', 'ap-36k']):
            return 'Animal Pose Estimation'
        if _any(t, ['few-shot', 'novel class']):
            return 'Few-shot Pose Estimation'
        return '(general)'

    # ------------------------------------------------------------------
    # 13. 3D Human Pose & Shape Estimation
    # ------------------------------------------------------------------
    if order == '3D Human Pose & Shape Estimation':
        if _any(t, ['video', 'temporal', 'motion', 'sequence']):
            return 'Video 3D Pose Estimation'
        if _any(t, ['expressive', 'hand face', 'whole-body', 'smplx']):
            return 'Expressive Whole-body Estimation'
        if _any(t, ['in-the-wild', 'occlusion', 'crowded']):
            return 'In-the-wild 3D Pose'
        if _any(t, ['one-shot', 'few-shot', 'generaliz']):
            return 'Few-shot 3D Pose'
        if _any(t, ['mesh', 'surface', 'smpl', 'parametric']):
            return 'Human Mesh Recovery'
        return '(general)'

    # ------------------------------------------------------------------
    # 14. Face Recognition & Verification
    # ------------------------------------------------------------------
    if order == 'Face Recognition & Verification':
        if _any(t, ['arcface', 'cosface', 'sphereface', 'am-softmax', 'circle loss', 'adaface']):
            return 'Margin-based Face Recognition'
        if _any(t, ['anti-spoof', 'liveness', 'presentation attack', 'face forgery']):
            return 'Face Anti-spoofing & Liveness'
        if _any(t, ['deepfake', 'forgery detect', 'manipulat detect']):
            return 'Deepfake Detection'
        if _any(t, ['occlusion', 'partial face', 'masked face']):
            return 'Occluded Face Recognition'
        if _any(t, ['low-resolution', 'low resol', 'small face', 'surveillance']):
            return 'Low-resolution Face Recognition'
        if _any(t, ['privacy', 'differential', 'federated']):
            return 'Privacy-preserving Face Recognition'
        return '(general)'

    # ------------------------------------------------------------------
    # 15. Optical Flow Estimation
    # ------------------------------------------------------------------
    if order == 'Optical Flow Estimation':
        if _any(t, ['self-supervised', 'unsupervised', 'self-supervis']):
            return 'Self-supervised Optical Flow'
        if _any(t, ['raft', 'gma ', 'flowformer', 'unimatch']):
            return 'Iterative / Transformer Flow'
        if _any(t, ['real-time', 'efficient', 'fast', 'lightweight']):
            return 'Efficient Optical Flow'
        if _any(t, ['event camera', 'event-based', 'neuromorphic']):
            return 'Event-based Optical Flow'
        if _any(t, ['large displacement', 'long-range']):
            return 'Large-displacement Flow'
        return '(general)'

    # ------------------------------------------------------------------
    # 16. Multi-Object Tracking
    # ------------------------------------------------------------------
    if order == 'Multi-Object Tracking':
        if _any(t, ['transformer', 'trackformer', 'motr ', 'transtrack']):
            return 'Transformer-based MOT'
        if _any(t, ['association', 'data association', 'hungarian', 'gnn']):
            return 'Association-based MOT'
        if _any(t, ['lidar', 'point cloud', '3d track', 'bev track']):
            return '3D Multi-object Tracking'
        if _any(t, ['detection-free', 'track-then-detect']):
            return 'Detection-free MOT'
        if _any(t, ['online', 'real-time', 'bytetrack', 'strongsort', 'sort']):
            return 'Online MOT'
        return '(general)'

    # ------------------------------------------------------------------
    # 17. Single Object Tracking
    # ------------------------------------------------------------------
    if order == 'Single Object Tracking':
        if _any(t, ['siamese', 'siamrpn', 'siammask', 'dimp']):
            return 'Siamese Network Tracking'
        if _any(t, ['transformer', 'ostrack', 'mixformer', 'stark', 'transt']):
            return 'Transformer-based SOT'
        if _any(t, ['long-term', 're-detect', 're-detection', 'global']):
            return 'Long-term Tracking'
        if _any(t, ['rgb-d', 'depth', 'infrared', 'thermal']):
            return 'Multi-modal SOT'
        if _any(t, ['efficient', 'lightweight', 'real-time', 'mobile']):
            return 'Efficient SOT'
        return '(general)'

    # ------------------------------------------------------------------
    # 18. Contrastive Self-supervised Learning
    #     The previous Order "Video Self-supervised Learning" (under the
    #     "Self-supervised Learning" Class) was demoted into the
    #     "Video Self-supervised Learning" Genus below.
    # ------------------------------------------------------------------
    if order == 'Contrastive Self-supervised Learning':
        # Video SSL is a variant of contrastive SSL applied to video; check
        # before family-name rules so video-specific titles still resolve here.
        if _any(t, ['video', 'temporal', 'video self-supervised',
                    'video pretrain', 'video contrastive']):
            return 'Video Self-supervised Learning'
        if _any(t, ['moco', 'momentum contrast']):
            return 'MoCo Family'
        if _any(t, ['simclr', 'nt-xent']):
            return 'SimCLR Family'
        if _any(t, ['byol', 'simsiam', 'non-contrastive', 'bootstrap']):
            return 'Non-contrastive SSL'
        if _any(t, ['dino', 'dinov2', 'self-distil']):
            return 'DINO / Self-distillation'
        if _any(t, ['barlow', 'vicreg', 'redundancy']):
            return 'Redundancy-reduction SSL'
        if _any(t, ['multimodal', 'audio-visual', 'video-text', 'cross-modal']):
            return 'Multimodal Contrastive Learning'
        if _any(t, ['dense', 'pixel-level', 'region-level', 'local contrast']):
            return 'Dense / Pixel-level Contrastive'
        return '(general)'

    # ------------------------------------------------------------------
    # 19. Masked Autoencoders (MAE)
    # ------------------------------------------------------------------
    if order == 'Masked Autoencoders (MAE)':
        if _any(t, ['video', 'temporal', 'videomae', 'video masked']):
            return 'Video MAE'
        if _any(t, ['multimodal', 'point cloud', 'depth', 'multi-modal mae']):
            return 'Multimodal MAE'
        if _any(t, ['efficient', 'fast mae', 'sparse mae', 'lightweight']):
            return 'Efficient MAE'
        if _any(t, ['medical', 'clinical', 'pathology']):
            return 'Medical MAE'
        if _any(t, ['beit', 'discrete token', 'dVAE', 'tokenizer']):
            return 'Discrete Token Prediction (BEiT)'
        return '(general)'

    # ------------------------------------------------------------------
    # 20. Domain Adaptation
    # ------------------------------------------------------------------
    if order == 'Domain Adaptation':
        if _any(t, ['semantic segment', 'urban', 'synthetic']):
            return 'DA for Segmentation'
        if _any(t, ['object detect', 'detection']):
            return 'DA for Detection'
        if _any(t, ['source-free', 'black-box', 'source-data-free']):
            return 'Source-free DA'
        if _any(t, ['test-time', 'online adapt', 'tta']):
            return 'Test-time Adaptation'
        if _any(t, ['multi-source', 'multiple source domain']):
            return 'Multi-source DA'
        if _any(t, ['adversarial', 'dann', 'mcd ', 'mdd']):
            return 'Adversarial Domain Adaptation'
        return '(general)'

    # ------------------------------------------------------------------
    # 21. Knowledge Distillation
    # ------------------------------------------------------------------
    if order == 'Knowledge Distillation':
        if _any(t, ['feature', 'intermediate', 'activation', 'hint']):
            return 'Feature-based KD'
        if _any(t, ['online', 'mutual', 'born again', 'collaborative']):
            return 'Online KD'
        if _any(t, ['data-free', 'generator', 'synthetic data kd']):
            return 'Data-free KD'
        if _any(t, ['contrastive', 'relational', 'rkd', 'crd']):
            return 'Contrastive / Relational KD'
        if _any(t, ['segment', 'detect', 'dense predict', 'dense kd']):
            return 'KD for Dense Prediction'
        if _any(t, ['large language', 'vlm', 'vision-language', 'llm']):
            return 'LLM / VLM Knowledge Distillation'
        return '(general)'

    # ------------------------------------------------------------------
    # 22. Image Restoration
    # ------------------------------------------------------------------
    if order == 'Image Restoration':
        if _any(t, ['diffusion', 'score-based', 'denoising diffusion']):
            return 'Diffusion-based Restoration'
        if _any(t, ['transformer', 'restormer', 'uformer', 'swinir']):
            return 'Transformer-based Restoration'
        if _any(t, ['blind', 'unknown degradat', 'degradation-blind']):
            return 'Blind Image Restoration'
        if _any(t, ['all-in-one', 'unified', 'multi-weather', 'multi-degradat']):
            return 'All-in-one Restoration'
        if _any(t, ['real-world', 'in-the-wild', 'practical']):
            return 'Real-world Image Restoration'
        return '(general)'

    # ------------------------------------------------------------------
    # 23. Single Image Super-Resolution
    # ------------------------------------------------------------------
    if order == 'Single Image Super-Resolution':
        if _any(t, ['blind', 'real-world', 'unknown degradat', 'real-esrgan', 'bsrgan']):
            return 'Blind / Real-world SR'
        if _any(t, ['diffusion', 'score-based']):
            return 'Diffusion-based SR'
        if _any(t, ['transformer', 'swinir', 'hat ', 'edat']):
            return 'Transformer-based SR'
        if _any(t, ['gan', 'perceptual', 'esrgan', 'esrnet']):
            return 'Perceptual / GAN-based SR'
        if _any(t, ['arbitrary-scale', 'implicit neural', 'liif']):
            return 'Implicit / Arbitrary-scale SR'
        if _any(t, ['lightweight', 'efficient', 'fast', 'mobile']):
            return 'Lightweight SR'
        return '(general)'

    # ------------------------------------------------------------------
    # 24. Vision Transformers (ViT)
    #     The previous Order "Efficient Vision Transformers" was demoted
    #     into the "Efficient ViT" Genus below — it's the same architecture
    #     family, just compressed/accelerated.
    # ------------------------------------------------------------------
    if order == 'Vision Transformers (ViT)':
        if _any(t, ['efficient', 'lite vit', 'mobile vit', 'efficientvit',
                    'fastvit', 'lightweight', 'compact vit']):
            return 'Efficient ViT'
        if _any(t, ['deit', 'training strategy', 'distillation vit', 'knowledge dist vit']):
            return 'ViT Training & Distillation'
        if _any(t, ['patch', 'token', 'token pruning', 'token merging']):
            return 'Token / Patch Design'
        if _any(t, ['scale', 'scaling', 'large-scale vit', 'huge vit', 'giant vit']):
            return 'Scaling ViT'
        if _any(t, ['pos embed', 'positional encoding', 'relative position']):
            return 'Positional Encoding in ViT'
        return '(general)'

    # ------------------------------------------------------------------
    # 25. Hierarchical Vision Transformers
    # ------------------------------------------------------------------
    if order == 'Hierarchical Vision Transformers':
        if _any(t, ['swin transformer', 'swintransformer', 'swin-t', 'swin-s',
                    'swin-b', 'swin-l', 'swin-v2', 'swin v2', ' swin ']):
            return 'Swin Transformer'
        if _any(t, ['pvt', 'pyramid vision transformer']):
            return 'PVT Family'
        if _any(t, ['efficient', 'lightweight', 'mobile']):
            return 'Efficient Hierarchical ViT'
        if _any(t, ['local window', 'window attention', 'shifted window']):
            return 'Window Attention Transformers'
        return '(general)'

    # ------------------------------------------------------------------
    # 26. Few-shot Learning
    # ------------------------------------------------------------------
    if order == 'Few-shot Learning':
        if _any(t, ['prototypical', 'prototype', 'proto-net']):
            return 'Prototypical Networks'
        if _any(t, ['matching network', 'relation network', 'maml', 'anil ']):
            return 'Meta-learning Methods'
        if _any(t, ['inductive', 'transductive', 'semi-transductive']):
            return 'Transductive Few-shot Learning'
        if _any(t, ['cross-domain', 'domain generaliz few-shot']):
            return 'Cross-domain Few-shot'
        if _any(t, ['generalized few-shot', 'gfsl']):
            return 'Generalized Few-shot Learning'
        if _any(t, ['pretrain', 'foundation model', 'large-scale pretrain']):
            return 'Pretrained Model Few-shot'
        return '(general)'

    # ------------------------------------------------------------------
    # 27. Adversarial Attacks
    # ------------------------------------------------------------------
    if order in ('Adversarial Attacks', 'Physical & Universal Adversarial Attacks',
                 'Transferable Adversarial Attacks'):
        if _any(t, ['physical', 'stop sign', 'patch attack', 'printed']):
            return 'Physical Adversarial Attacks'
        if _any(t, ['universal', 'input-agnostic', 'untargeted universal']):
            return 'Universal Adversarial Perturbations'
        if _any(t, ['transfer', 'black-box', 'query-based', 'decision-based']):
            return 'Transferable / Black-box Attacks'
        if _any(t, ['semantic', 'attribute', 'face adversarial']):
            return 'Semantic Adversarial Attacks'
        if _any(t, ['certif', 'pgd', 'apgd', 'autoattack', 'fgsm']):
            return 'White-box Gradient Attacks'
        if _any(t, ['detection', 'object detect', 'segment', 'track']):
            return 'Adversarial Attacks on Dense Tasks'
        return '(general)'

    # ------------------------------------------------------------------
    # 28. Medical Image Segmentation
    # ------------------------------------------------------------------
    if order == 'Medical Image Segmentation':
        if _any(t, ['transformer', 'swin unet', 'transunet', 'medt', 'medformer',
                    'vit-based', 'vision transformer medical']):
            return 'Transformer-based Medical Segmentation'
        if _any(t, ['u-net', 'unet', 'u net']):
            return 'U-Net Family'
        if _any(t, ['semi-supervised', 'few-shot', 'label-efficient']):
            return 'Label-efficient Medical Segmentation'
        if _any(t, ['segment anything', ' sam ', 'foundation model']):
            return 'Foundation Model for Medical Segmentation'
        if _any(t, ['brain', 'mri segment', 'white matter', 'cortical']):
            return 'Brain MRI Segmentation'
        if _any(t, ['lung', 'chest', 'nodule', 'airway']):
            return 'Lung & Chest Segmentation'
        if _any(t, ['polyp', 'colon', 'endoscop']):
            return 'Polyp & Endoscopy Segmentation'
        if _any(t, ['cell', 'nuclei', 'histolog', 'pathology']):
            return 'Cell & Histology Segmentation'
        return '(general)'

    # ------------------------------------------------------------------
    # 29. Point Cloud Classification
    # ------------------------------------------------------------------
    if order == 'Point Cloud Classification':
        if _any(t, ['transformer', 'point transformer', 'pct', 'pointbert']):
            return 'Transformer-based Point Cloud Classification'
        if _any(t, ['graph', 'dgcnn', 'dynamic graph']):
            return 'Graph-based Point Cloud Classification'
        if _any(t, ['self-supervised', 'pretrain', 'masked point', 'point mae']):
            return 'Self-supervised Point Cloud Learning'
        if _any(t, ['few-shot', 'zero-shot', 'generaliz']):
            return 'Few-shot Point Cloud Recognition'
        if _any(t, ['scanobjectnn', 'modelnet', 'shapenet']):
            return 'Benchmark Point Cloud Classification'
        return '(general)'

    # ------------------------------------------------------------------
    # 30. Point Cloud Object Detection
    # ------------------------------------------------------------------
    if order == 'Point Cloud Object Detection':
        if _any(t, ['anchor-free', 'center-based', 'centerpoint', 'ia-ssd']):
            return 'Anchor-free 3D Detection'
        if _any(t, ['transformer', 'detr 3d', 'petr', 'point transformer det']):
            return 'Transformer-based 3D Detection'
        if _any(t, ['bev', "bird's eye", 'voxel bev']):
            return 'BEV-based 3D Detection'
        if _any(t, ['multi-modal', 'lidar-camera', 'sensor fusion detect']):
            return 'Multi-modal 3D Detection'
        if _any(t, ['semi-supervised', 'weakly', 'few-shot 3d']):
            return 'Label-efficient 3D Detection'
        return '(general)'

    # ------------------------------------------------------------------
    # 31. Graph Neural Networks
    #     The previous Order "Spatial-Temporal Graph Networks" was demoted
    #     into the "Spatial-Temporal GNN" Genus below — same backbone, just
    #     with a temporal axis.
    # ------------------------------------------------------------------
    if order == 'Graph Neural Networks':
        if _any(t, ['spatial-temporal', 'st-gcn', 'dynamic graph',
                    'temporal graph', 'spatio-temporal graph']):
            return 'Spatial-Temporal GNN'
        if _any(t, ['graph attention', ' gat ', 'attention-based gnn']):
            return 'Graph Attention Networks'
        if _any(t, ['graph convol', ' gcn ']):
            return 'Graph Convolutional Networks'
        if _any(t, ['graph transformer']):
            return 'Graph Transformers'
        if _any(t, ['equivariant graph', 'message passing']):
            return 'Message Passing Neural Networks'
        if _any(t, ['scalable', 'large-scale graph', 'efficient graph']):
            return 'Scalable / Efficient GNNs'
        return '(general)'

    # ------------------------------------------------------------------
    # 32. General 3D Vision (catch-all under 3D Scene Understanding Class)
    #     Order renamed from "3D Scene Understanding" to avoid the
    #     Class==Order naming clash.
    # ------------------------------------------------------------------
    if order == 'General 3D Vision':
        if _any(t, ['3d feature', 'feature 3d', '3d descriptor']):
            return '3D Feature Learning'
        if _any(t, ['3d understand', 'scene understand']):
            return '3D Scene Reasoning'
        if _any(t, ['3d human', 'human 3d', 'avatar']):
            return '3D Human Modeling'
        if _any(t, ['event camera', 'event-based', 'event stream', 'dvs']):
            return 'Event-based 3D Vision'
        if _any(t, ['dense 3d', 'volumetric', 'tsdf', 'voxel']):
            return 'Volumetric 3D Vision'
        return '(general)'

    # ------------------------------------------------------------------
    # 33. 3D Shape Analysis
    # ------------------------------------------------------------------
    if order == '3D Shape Analysis':
        if _any(t, ['shape correspondence', 'shape match']):
            return 'Shape Correspondence & Matching'
        if _any(t, ['shape retriev', 'shape descriptor']):
            return 'Shape Retrieval & Description'
        if _any(t, ['shape from shading', 'shape recovery', 'shape from']):
            return 'Shape Recovery'
        if _any(t, ['morphable model', '3dmm']):
            return 'Morphable Shape Models'
        if _any(t, ['shape model', 'shape analysis']):
            return 'Generic Shape Modeling'
        return '(general)'

    # ------------------------------------------------------------------
    # 34. AR/VR Scene Understanding
    # ------------------------------------------------------------------
    if order == 'AR/VR Scene Understanding':
        if _any(t, ['augmented reality']):
            return 'Augmented Reality Scene Understanding'
        if _any(t, ['mixed reality']):
            return 'Mixed Reality Scene Understanding'
        if _any(t, ['virtual reality', ' vr ', 'vr headset']):
            return 'Virtual Reality Scene Understanding'
        return '(general)'

    # ------------------------------------------------------------------
    # 35. General Image Segmentation (catch-all under Image Segmentation Class)
    #     Order renamed from "Image Segmentation" to avoid Class==Order clash.
    # ------------------------------------------------------------------
    if order == 'General Image Segmentation':
        if _any(t, ['boundary', 'edge segment']):
            return 'Boundary-aware Segmentation'
        if _any(t, ['real-time', 'efficient', 'lightweight', 'fast']):
            return 'Efficient Segmentation'
        if _any(t, ['transformer', 'attention']):
            return 'Transformer-based Segmentation'
        if _any(t, ['multi-class', 'multi-label']):
            return 'Multi-class Segmentation'
        if _any(t, ['encoder-decoder', 'fcn ', 'unet', 'u-net']):
            return 'Encoder-Decoder Segmentation'
        return '(general)'

    # ------------------------------------------------------------------
    # 36. General VQA (catch-all under Visual Question Answering Class)
    #     Order renamed from "Visual Question Answering" to avoid clash.
    # ------------------------------------------------------------------
    if order == 'General VQA':
        if _any(t, ['compositional', 'gqa']):
            return 'Compositional VQA'
        if _any(t, ['attention', 'co-attention']):
            return 'Attention-based VQA'
        if _any(t, ['transformer', 'vit']):
            return 'Transformer-based VQA'
        if _any(t, ['robust', 'bias', 'shortcut']):
            return 'Robust / Debiased VQA'
        if _any(t, ['open-ended', 'free-form']):
            return 'Open-ended VQA'
        return '(general)'

    # ------------------------------------------------------------------
    # Default
    # ------------------------------------------------------------------
    return '(general)'


# ===========================================================================
# Quick smoke test
# ===========================================================================

if __name__ == '__main__':
    from classify import classify

    test_cases = [
        ("Self-supervised Monocular Depth Estimation via Geometry-aware Features",
         'Self-supervised Depth'),
        ("Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
         'Efficient NeRF'),
        ("Compact Gaussian Splatting for Efficient Scene Representation",
         'Compact Gaussian Splatting'),
        ("Denoising Diffusion Probabilistic Models",
         '(general)'),
        ("ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
         'Margin-based Face Recognition'),
        ("RAFT: Recurrent All-Pairs Field Transforms for Optical Flow",
         'Iterative / Transformer Flow'),
        ("ByteTrack: Multi Object Tracking by Associating Every Detection Box",
         'Online MOT'),
        ("Masked Autoencoders Are Scalable Vision Learners",
         '(general)'),
        ("SwinTransformer: Hierarchical Vision Transformer using Shifted Windows",
         'Swin Transformer'),
        ("TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation",
         'Transformer-based Medical Segmentation'),
    ]

    passed = 0
    failed = 0
    print("=" * 70)
    print("Genus Rules Smoke Test")
    print("=" * 70)
    for title, expected_genus in test_cases:
        phylum, class_, order = classify(title)
        genus = assign_genus(phylum, class_, order, title)
        ok = genus == expected_genus
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {title[:50]:<50}")
        if not ok:
            print(f"       Order    : {order}")
            print(f"       Expected : {expected_genus}")
            print(f"       Got      : {genus}")
    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")
    print("=" * 70)
