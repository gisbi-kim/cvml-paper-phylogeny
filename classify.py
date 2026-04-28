"""
CVML Phylogenetic Taxonomy Classifier
======================================
Maps 112,183 paper titles into 4-level taxonomy:
  Phylum > Class > Order > Genus

14 Phyla, ~110 Classes, ~380 Orders.
Rules ordered by specificity. Earlier rules win.
Patterns encode SEMANTIC equivalence (synonym clusters).

Usage:
    from classify import classify
    phylum, class_, order = classify("Deep Residual Learning for Image Recognition")
"""

from __future__ import annotations
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def t_low(title: str) -> str:
    """Lowercase + add leading/trailing space for whole-word boundary matching."""
    return ' ' + title.lower() + ' '


def _any(t: str, kws: list[str]) -> bool:
    """Return True if any keyword is found in the lowercased title string."""
    return any(kw in t for kw in kws)


# ---------------------------------------------------------------------------
# Synonym clusters — 3D Vision
# ---------------------------------------------------------------------------

NERF = [
    'nerf', 'neural radiance', 'neural implicit', 'implicit neural',
    'occupancy network', 'signed distance', 'nerf-based', 'nerfs',
    'instant ngp', 'mip-nerf', 'block-nerf', 'deformable nerf',
    'dynamic nerf', 'nerf in the wild',
    'instant neural graphics', 'neural graphics primitive',
    'hash encoding 3d', 'radiance field', 'neural field',
    'continuous visual', 'scene representation network',
    'implicit surface', 'neural 3d', 'neural render',
    'volumetric render', 'differentiable render', 'inverse render',
    'neural scene', 'scene neural',
]
GAUSSIAN_SPLAT = [
    'gaussian splatting', '3d gaussian', 'gaussian splat',
    '4d gaussian', '2d gaussian splatting', 'gaussian avatar',
]
DEPTH_KW = [
    'depth estimation', 'depth prediction', 'monocular depth',
    'depth completion', 'depth map', 'depth from', 'depth super-resol',
    'depth upsampl', 'depth normal', 'dense depth',
]
STEREO_KW = [
    'stereo matching', 'stereo vision', 'stereo depth',
    'disparity estimation', 'binocular', 'stereo network',
    'stereo correspondence', 'active stereo', 'structured light',
    'time-of-flight', 'tof depth',
]
POINT_CLOUD = [
    'point cloud', 'pointcloud', 'point-cloud',
    'lidar', '3d point', 'range scan', 'range image',
    'point set', 'point-based', 'sparse 3d',
]
SFM = [
    'structure from motion', 'sfm', 'multi-view stereo',
    'visual localization', 'camera pose', 'bundle adjustment',
    'simultaneous localization', 'slam', 'place recognition',
    'visual odometry', 'relocalization', 'map-based localization',
    'scene coordinate', 'neural slam',
    'camera calibrat', 'camera model', 'epipolar',
    'homography', 'fundamental matrix', 'essential matrix',
    'from-motion', 'ego-motion', 'egomotion',
]
SCENE_3D = [
    'scene reconstruction', '3d reconstruction', '3d scene',
    'volumetric reconstruction', 'indoor 3d', 'outdoor 3d',
    'dense reconstruction', 'novel view synthesis', 'view synthesis',
    'light field', 'image-based rendering', '3d representation',
    'multi-view reconstruction', 'surface reconstruction',
    'truncated signed distance', 'tsdf', 'voxel grid',
    'shape reconstruct', 'dense 3d', '3d model', '3d shape',
    'range image', 'depth image fusion', 'voxel',
    '3d understand', '3d feature', '3d visual',
    '3d morphable', '3dmm', 'morphable model', 'surface normal',
    '3d avatar', 'human avatar 3d', 'augmented reality',
    'mixed reality', 'virtual reality 3d',
]

# ---------------------------------------------------------------------------
# Generative Models
# ---------------------------------------------------------------------------

GAN = [
    'generative adversarial', ' gan ', 'gans ', 'adversarial network',
    'image synthesis', 'image generation', 'photo-realistic',
    'image-to-image translation', 'conditional image generation',
    'style transfer', 'stylegan', 'pix2pix', 'cyclegan',
    'video generation', 'video synthesis', 'face synthesis',
    'face generation', 'face swap', 'deepfake generation',
    'talking head', 'talking face', 'neural talking',
]
DIFFUSION = [
    'diffusion model', 'diffusion-based', 'score matching',
    'denoising diffusion', 'ddpm', 'score-based', 'flow matching',
    'stable diffusion', 'latent diffusion', 'rectified flow',
    'consistency model', 'stochastic differential equation',
    'sde-based', 'classifier-free guidance', 'cfg guidance',
]
TEXTTOIMAG = [
    'text-to-image', 'text to image', 'text-driven image',
    'text-guided', 'dall-e', 'imagen', 'parti ',
    'text-guided synthesis', 'text-driven synthesis',
]
VAE_KW = [
    'variational autoencoder', ' vae ', 'vae-based',
    'variational inference', 'latent variable model',
    'discrete vae', 'codebook',
]
IMG_EDIT = [
    'image editing', 'image manipulation', 'image inpainting',
    'image outpainting', 'image completion', 'image blending',
    'image harmonization', 'visual editing', 'photo editing',
    'drag-based edit', 'instructpix2pix', 'instruct-pix2pix',
    'instructed image', 'instruction-based edit',
]
VIDEO_GEN = [
    'video diffusion', 'video generation', 'text-to-video',
    'text to video', 'video editing', 'video inpainting',
    'sora ', 'video synthesis',
]

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

SEGMENT = [
    'semantic segmenta', 'instance segmenta', 'panoptic segmenta',
    'image segmenta', 'scene parsing', 'pixel-wise classif',
    'fcn ', 'deeplab', 'mask r-cnn', 'maskrcnn',
    'segment anything', ' sam ', 'interactive segmen',
    'mask segmenta', 'pixel classif', 'pixel labeling',
    'semantic parsing', 'open-vocabulary segmenta', 'open vocabulary segmenta',
    'referring segmenta', 'open-set segmenta',
]
VID_SEG = [
    'video segmenta', 'video object segmen', 'video instance',
    'video panoptic', 'vos ', 'davis benchmark',
]
MED_SEG = [
    'medical segmenta', 'organ segmenta', 'lesion segmenta',
    'cell segmenta', 'tissue segmenta', 'gland segmenta',
    'polyp segmenta', 'skin lesion segmenta', 'retinal segmenta',
]

# ---------------------------------------------------------------------------
# Object Detection
# ---------------------------------------------------------------------------

DETECT = [
    'object detect', 'object recogni', 'bounding box',
    'region proposal', 'faster r-cnn', 'faster-rcnn', 'rcnn',
    'yolo', 'yolov', ' detr ', 'deformable detr',
    ' ssd ', 'focal loss detect', 'fcos ',
    'centernet', 'anchor-free detect', 'anchor-based detect',
    'one-stage detect', 'two-stage detect',
    'open-set detect', 'open-vocabulary detect',
    'weakly supervised detect', 'semi-supervised detect',
]
DETECT3D = [
    '3d object detect', 'lidar detect', 'point cloud detect',
    'bev detect', "bird's eye view detect", 'bird-eye detect',
    '3d bounding box', 'voxel-based detect', 'range-based detect',
]
GROUNDING = [
    'visual grounding', 'phrase grounding', 'referring express',
    'grounding dino', 'open-vocabulary grounding',
    'region-level understand', 'dense predict',
]
SALIENT = [
    'salient object', 'saliency detect', 'eye fixation',
    'visual saliency', 'salient region',
]
SMALL_OBJ = [
    'small object detect', 'tiny object detect',
    'small target detect', 'drone detect',
]

# ---------------------------------------------------------------------------
# Human-centric
# ---------------------------------------------------------------------------

FACE = [
    'face recogni', 'face detect', 'face verif', 'face identif',
    'facial recogni', 'face align', 'face anti-spoof', 'deepfake detect',
    'face attribute', 'face age', 'face super-resol', 'face hallucin',
    'face restoration', 'face enhance', 'face image',
    'arcface', 'sphereface', 'cosface', 'face liveness',
    'age estimat', 'age prediction', 'apparent age',
    'face shape', 'face 3d', '3d face',
    'face in the wild', 'face under', 'face network',
]
FACE_GEN = [
    'face synthesis', 'face generation', 'face swap',
    'face reenact', 'face animat', 'face manipulat',
    'face editing',
]
POSE = [
    'human pose', 'pose estimat', '2d pose', '3d pose',
    'skeleton estimat', 'body keypoint', 'human body model',
    'smpl ', 'human mesh', 'mocap', 'multi-person pose',
    'whole-body pose', 'expressive body', 'human body estimat',
    'hand-object interact',
]
REID = [
    'person re-identif', 'person reid', 'pedestrian re-id',
    'vehicle re-id', 're-identification', ' reid ',
    'occluded re-id', 'cloth-changing reid', 'text-to-person',
]
GESTURE = [
    'gesture recogn', 'hand gesture', 'sign language', 'hand pose',
    'hand track', 'hand shape', 'hand mesh', 'fingertip',
]
CROWD = [
    'crowd count', 'crowd density', 'pedestrian detect',
    'crowd estimat', 'crowd localiz', 'crowd scene',
    'density map', 'counting network', 'people count',
    'head detect', 'head count',
]
EMOTION = [
    'emotion recogni', 'facial expression', 'affect recogni',
    'sentiment recogni', 'facial action unit', 'expression recogni',
]
GAZE = [
    'gaze estimat', 'eye gaze', 'gaze predict',
    'gaze redirect', 'attention map human',
]
ACTIVITY = [
    'human activity', 'skeletal action', 'skeleton-based action',
    'graph-based action', 'human interaction recogni',
]

# ---------------------------------------------------------------------------
# Vision-Language & Multimodal
# ---------------------------------------------------------------------------

VQA = [
    'visual question', ' vqa', 'visual answer',
    'visual reasoning', 'visual commonsense', 'visual inferenc',
    'gqa ', 'vqa2', 'ok-vqa', 'visual-spatial reason',
]
CAPTION = [
    'image caption', 'visual caption', 'dense caption',
    'image descri', 'video caption', 'visual descript',
    'referring caption',
]
CLIP_KW = [
    ' clip ', 'clip:', 'clip-', 'contrastive language-image', 'clip-based',
    'language-image pretrain', 'blip', 'blip-2', 'blip2',
    'align and prompt', 'language-vision pretrain',
]
VLM = [
    'vision-language model', 'large vision-language', 'visual instruction',
    'llava', 'visual llm', 'gpt-4v', 'vision language pretrain',
    'visual pretrain', 'vision foundation model',
    'multimodal language model', 'multimodal large language',
    'flamingo', 'minigpt', 'qwen-vl', 'intern vl',
    'visual chat', 'vision chat',
]
CROSSMODAL = [
    'cross-modal', 'image-text', 'text-image retrieval',
    'visual-text', 'multimodal fusion', 'visual bert',
    'vilbert', 'uniter', 'oscar ', 'vinvl', 'albef',
    'multimodal align', 'audio-visual', 'video-text',
    'video question', 'video-language',
    'sentence matching', 'image and sentence', 'vision and language',
    'language and vision', 'visual language', 'language navigation',
    'visual navigation instruction', 'vision-language navigation',
]
GROUNDING_VL = [
    'text-guided detect', 'open-vocabulary', 'language-guided',
    'visual grounding language',
]

# ---------------------------------------------------------------------------
# Video & Motion
# ---------------------------------------------------------------------------

ACTION_RECOG = [
    'action recogni', 'activity recogni', 'action classif',
    'video classif', 'sport recogni', 'video understand',
    'temporal segment', 'two-stream', 'slowfast', 'i3d ',
    'video action', 'fine-grained action', 'untrimmed video',
]
ACTION_DETECT = [
    'temporal action', 'action detect', 'action segment',
    'activity detect', 'event detect', 'action localiz',
    'temporal grounding', 'moment retrieval', 'natural language grounding video',
]
OPTFLOW = [
    'optical flow', 'motion estimat', 'scene flow',
    'motion field', 'video flow', 'pwcnet', 'raft flow',
    'flownet', 'motion compensat',
]
VIDEO_PREDICT = [
    'video predict', 'future frame', 'video forecast',
    'temporal predict', 'video extrapolat', 'video anticip',
]
TRACKING = [
    'object track', 'visual track', 'single object track',
    'multi-object track', 'multi-target track', 'video track',
    'long-term track', 'siamese track', ' mot ', ' sot ',
    'deepsort', 'bytetrack', 'tracking-by-detect',
    'multi-pedestrian track', 'point track',
    'multi object track', 'object tracker', 'online track',
    'track every', 'track anything', 'tracker ',
]
VIDEO_REPR = [
    'video represent', 'video self-supervised', 'video pretrain',
    'temporal model', 'video foundation', 'video transformer',
    'video masked autoencode',
]
TRAJECTORY = [
    'trajectory predict', 'trajectory forecast', 'trajectory estimat',
    'motion forecast', 'future trajectory', 'path predict',
    'pedestrian trajectory', 'vehicle trajectory', 'agent trajectory',
    'trajectory plan', 'trajectory model',
]

# ---------------------------------------------------------------------------
# Low-level Vision
# ---------------------------------------------------------------------------

SUPERRES = [
    'super-resolution', 'super resolution', 'image upscal',
    'image upsampl', 'sr network', 'single image sr',
    'blind sr', 'real-world sr', 'real-esrgan',
    'face super-resol', 'video super-resol', 'arbitrary-scale',
]
RESTORE = [
    'image restor', 'image denois', 'image deblur',
    'blind deconvol', 'image derain', 'image dehaz',
    'image artifact', 'image noise', 'noise remov',
    'jpeg artifact', 'image quality enhance', 'image deraining',
    'image defogging', 'image desnow', 'all-in-one restor',
    'adverse weather', 'rain streak', 'rain remov',
    'derain', 'dehaz', 'desnow', 'defog',
    'video denois', 'video derain', 'video dehaz',
    'deraining', 'dehazing', 'denoising', 'deblurring',
    'image blind', 'image clean',
]
ENHANCE = [
    'low-light', 'image enhance', 'hdr imaging', 'tone mapping',
    'image brightening', 'exposure correct', 'color correct',
    'color enhance', 'retinex', 'illumination estimat',
    'low light', 'dark image', 'nighttime image',
    'image illuminat', 'image color', 'underwater image',
    'color constancy', 'white balance',
]
IMG_COMPRESS = [
    'image compress', 'learned compress', 'neural compress',
    'video compress', 'rate-distortion', ' codec ',
    'end-to-end compress', 'perceptual compress', 'flow-based compress',
]
IMG_QUALITY = [
    'image quality assess', 'no-reference quality', ' iqa ',
    'perceptual quality', 'image fidelity', 'blind image quality',
]
COMP_PHOTO = [
    'rolling shutter', 'computational photography', 'image stitch',
    'image mosaic', 'panorama', 'image fusion', 'exposure fusion',
    'bokeh', 'lens flare', 'reflection remov', 'demosaic', 'raw image',
    'image formation', 'point spread function', 'psf ', 'deconvolut',
    'blind deconvolut',
]

# ---------------------------------------------------------------------------
# Representation Learning
# ---------------------------------------------------------------------------

SSL = [
    'self-supervised', 'contrastive learn', 'contrastive represent',
    'self-supervised pretrain', 'momentum contrast', 'moco ',
    'simclr', 'byol ', ' dino ', 'barlow twins',
    'non-contrastive', 'self-distil', 'positive pair',
    'negative-free', 'w-mse', 'vicreg',
]
MAE = [
    'masked autoencoder', 'masked image model', ' mae ',
    'bert pretrain', 'masked pretrain', 'bert-style',
    'beit', 'ibot ', 'masked visual', 'image masked',
    'masked feature', 'data2vec',
]
DOMAIN_ADAPT = [
    'domain adaptation', 'domain shift', 'domain generaliz',
    'unsupervised domain', 'covariate shift', 'source-free adapt',
    'test-time adapt', 'tta ', 'domain randomiz', 'domain augment',
    'adversarial domain adapt',
]
TRANSFER = [
    'transfer learning', 'pre-trained model', 'pretrained model',
    'fine-tuning', 'fine-tune', 'prompt tuning', 'adapter tuning',
    'parameter-efficient', 'visual prompt', 'prefix tuning',
    'lora ', ' peft ',
]
FEW_SHOT_REP = [
    'few-shot learn', 'few shot learn', 'zero-shot learn',
    'zero shot learn', 'meta-learn', 'meta learn',
    'learning to learn', 'episodic train', 'prototypical network',
    'matching network', 'relation network few', 'generalized few-shot',
    'few-shot classif', 'zero-shot classif',
]
KG_EMBED = [
    'knowledge graph', 'knowledge distill embed', 'scene graph',
    'visual relation', 'object relation', 'relational learn',
]
SPARSE_REPR = [
    'sparse coding', 'sparse representation', 'dictionary learn',
    'dictionary atom', 'sparse autoencod', 'sparse network',
    'sparse attention', 'overcomplete dict', 'basis pursuit',
    'compressed sensing', 'compressive sensing', 'sparse signal',
    'subspace learn', 'subspace cluster', 'low-rank represent',
    'matrix factorization', 'tensor decomposit', 'tensor factori',
    'dimensionality reduc', 'principal component', ' pca ',
    'manifold embed', 'manifold repres', 'spectral embed',
]
CLUSTERING_KW = [
    'deep clustering', 'image clustering', 'visual clustering',
    'contrastive cluster', 'prototype cluster', 'unsupervised cluster',
    ' k-means', 'spectral cluster', 'cluster assign',
]

# ---------------------------------------------------------------------------
# Image Recognition & Retrieval
# ---------------------------------------------------------------------------

CLASSIF = [
    'image classif', 'visual classif', 'image recogni',
    'visual recogni', 'image categoriz', 'object categoriz',
    'multi-label classif', 'multi-class classif',
    'imagenet classif', 'visual categoriz',
    'pattern classif', 'pattern recogni',
    'target recogni', 'target classif',
    'object classif', 'shape classif',
    'deep classif', 'classif network',
    'visual category', 'category recogni',
]
FINEGRAINED = [
    'fine-grained recogni', 'fine-grained classif',
    'fine-grained visual', 'cub-200', 'stanford cars',
    'food recogni', 'plant recogni', 'bird recogni',
    'fine-grained categori', 'attribute-based classif',
]
RETRIEVAL = [
    'image retrieval', 'visual retrieval', 'visual search',
    'hash-based retrieval', 'hashing ', 'compact represent',
    'content-based retrieval', 'cbir', 'instance retrieval',
    'place retrieval', 'cross-modal retrieval',
]
METRIC = [
    'metric learning', 'distance metric', 'embedding learning',
    'siamese network', 'triplet loss', 'contrastive loss',
    'proxy loss', 'n-pair loss', 'deep metric',
    'embedding space', 'margin-based loss',
]
FEAT_MATCH = [
    'feature matching', 'keypoint match', 'local feature',
    'sift ', ' orb ', 'superpoint', 'superglue',
    'image match', 'descriptor match', 'loftr',
    'feature descriptor', 'patch descriptor',
]
SCENE_RECOG = [
    'scene recogni', 'scene classif', 'scene understand',
    'place recogni', 'geo-locali', 'geo-localization',
    'geolocali', 'indoor scene', 'room type',
]

# ---------------------------------------------------------------------------
# Deep Learning Architecture
# ---------------------------------------------------------------------------

CNN_ARCH = [
    'convolutional neural', 'depthwise convol', 'dilated convol',
    'residual network', 'resnet', 'densenet', 'mobilenet',
    'efficientnet', 'inception network', 'squeezenet', 'shufflenet',
    'inverted residual', 'strided convol', 'deformable convol',
    'dynamic convol',
]
VIT_ARCH = [
    'vision transformer', ' vit ', 'deit ', 'swin transformer',
    'swin-', ' pvt ', 'image worth', 'image is worth', 'patch embed',
    'tokens-to-token', 't2t-vit', 'crossvit', 'pyramid vision',
    'focal transformer', 'xcit ', 'cswin', 'twins transformer',
    'vit-b', 'vit-l', 'vit-h', 'vit-s', 'vitdet',
]
GNN_ARCH = [
    'graph neural', 'graph convol', ' gnn ', 'graph attention',
    'graph network', 'point-gnn', 'dynamic graph',
    'heterogeneous graph', 'knowledge graph network',
    'graph transformer', 'spatial-temporal graph',
]
NAS = [
    'neural architecture search', ' nas ', 'differentiable nas',
    'architecture search', 'once-for-all', 'network design space',
    'network architecture optim', 'darts ', 'proxyless',
    'single-path nas', 'auto-ml', 'hyperparameter optim',
]
ATTENTION = [
    'attention mechanism', 'self-attention', 'channel attention',
    'spatial attention', ' cbam ', 'squeeze-and-excit',
    'non-local means', 'non-local network', 'global context',
    'axial attention', 'multi-head attention', 'efficient attention',
    'linear attention', 'deformable attention',
    'attention is all you need', 'transformer attention',
    'cross-attention', 'co-attention',
    'non-local neural', 'non-local block',
]
MLP_ARCH = [
    'mlp-mixer', 'mlp mixer', ' gmlp ', 'vision mlp',
    'feedforward network', 'mlp for vision',
]
STATE_SPACE = [
    'state space model', 'mamba ', 'selective state',
    ' ssm ', 'linear recurrent', 'hippo ', 's4 model',
]
TRANSFORMER_GEN = [
    'transformer architecture', 'transformer for vision',
    'visual transformer', 'bert for vision',
    'neural network architecture', 'deep neural network design',
    'recurrent neural', 'lstm ', ' rnn ', 'gru ',
    'encoder decoder', 'encoder-decoder network',
    'generative model',
]

# ---------------------------------------------------------------------------
# Training & Learning Methods
# ---------------------------------------------------------------------------

OPTIM = [
    'stochastic gradient', ' sgd ', 'adam optimizer', 'adamw',
    'learning rate', 'lr schedule', 'warmup schedule', 'cosine anneal',
    'gradient descent', 'first-order optim', 'second-order optim',
    'sharpness-aware minim', 'sam optim', 'lookahead',
]
AUGMENT = [
    'data augmentation', 'mixup', 'cutout', 'cutmix', 'randaugment',
    'autoaugment', 'augmentation strategy', 'copy-paste augment',
    'trivialaugment', 'gridmask', 'style augment',
]
LONGTAIL = [
    'long-tail', 'long tail', 'class imbalance', 'imbalanced data',
    'tail class', 'class frequency', 'data imbalance',
    'head and tail', 'minority class',
]
NOISY_LBL = [
    'noisy label', 'label noise', 'learning with noise',
    'clean label', 'label correct', 'robust to noise label',
]
SEMISUP = [
    'semi-supervised', 'pseudo label', 'mean teacher',
    'self-training', 'label propagat', 'consistency regulariz',
    'fixmatch', 'flexmatch', 'comatch', 'umatch',
    'semi-supervised classif',
]
WEAKLY = [
    'weakly supervised', 'weak supervision', 'image-level label',
    'weakly-labeled', 'incomplete label', 'partial label',
    'multiple instance learn',
]
CURRICULUM = [
    'curriculum learn', 'self-paced learn', 'easy-to-hard',
    'progressive learn', 'difficulty-aware',
]
CONTINUAL = [
    'continual learn', 'incremental learn', 'catastrophic forget',
    'class-incremental', 'task-incremental', 'lifelong learn',
    'exemplar-free', 'replay method', 'elastic weight',
    'experience replay',
]
ACTIVE = [
    'active learn', 'query strateg', 'sample select', 'core-set',
    'informative sample', 'uncertainty sampling',
]
KD = [
    'knowledge distill', 'teacher-student', 'model distill',
    'offline distill', 'online distill', 'feature distill',
    'response-based distill', 'relation-based distill',
    'born again network', 'self-distil',
]

# ---------------------------------------------------------------------------
# Efficient & Robust ML
# ---------------------------------------------------------------------------

PRUNING = [
    'network pruning', 'weight pruning', 'filter pruning',
    'unstructured pruning', 'structured pruning', 'lottery ticket',
    'sparse network', 'dynamic sparse', 'magnitude pruning',
]
QUANT = [
    'network quantiz', 'model quantiz', 'post-training quantiz',
    'quantization-aware', 'binary neural', 'low-bit',
    'mixed-precision', 'weight quantiz', 'activation quantiz',
    'integer quantiz',
]
COMPRESS_ML = [
    'model compress', 'efficient inference', 'lightweight model',
    'efficient deep learn', 'network compress',
    'mobile deploy', 'edge deploy', 'hardware-aware',
]
ADV = [
    'adversarial example', 'adversarial attack', 'adversarial perturbat',
    'adversarial robust', 'adversarial training', 'pgd attack', ' fgsm ',
    'universal adversarial', 'physical adversarial', 'patch attack',
    'autoattack',
]
ADV_DEF = [
    'adversarial defense', 'certified robust', 'certified defense',
    'randomized smooth', 'input purif', 'adversarial purif',
]
OOD = [
    'out-of-distribution', 'ood detect', 'distribution shift',
    'open-set recogni', 'anomaly detect', 'novelty detect',
    'outlier detect', 'ood generalization', 'wild distribution',
]
UNCERTAINTY = [
    'uncertainty estimat', 'prediction uncertainty',
    'epistemic uncertainty', 'aleatoric uncertainty',
    'confidence calibrat', 'calibration', 'bayesian deep',
    'monte carlo dropout', 'deep ensemble',
]
FAIRNESS = [
    'fairness', ' bias ', 'debiasing', 'demographic parity',
    'equalized odds', 'algorithmic fairness', 'shortcut learn',
    'spurious correlat', 'dataset bias',
]
FEDERATED = [
    'federated learn', 'federated optim', 'communication-efficient',
    'distributed learn across', 'heterogeneous client',
    'federated aggregat', 'split learn',
]
XAI = [
    'explainab', 'interpretab', 'saliency map', 'grad-cam',
    'visual explanation', 'attribution method', 'feature importance',
    'class activation', 'lime ', 'shap ', 'network interpret',
]
PRIVACY = [
    'differential privacy', 'privacy-preserv', 'data privacy',
    'machine unlearn', 'membership inference', 'model inversion',
]

# ---------------------------------------------------------------------------
# Application Domains
# ---------------------------------------------------------------------------

MEDICAL = [
    'medical image', 'clinical imag', ' mri ', 'ct scan', 'x-ray',
    'chest x-ray', 'chest radiograph', 'pathology imag', 'histolog',
    'endoscop', 'colonoscop', 'fundus', 'retinal imag', 'ultrasound imag',
    'echocardiograph', 'brain segmenta', 'brain mri', 'tumor detect',
    'lesion detect', 'lesion segment', 'cell segmenta', 'microscopy',
    'cytology', 'dermatolog', 'skin cancer', 'diabetic retinopathy',
    'optic disc', 'optic cup', 'glaucoma', 'covid', 'lung segment',
    'liver segment', 'kidney segment', 'prostate segment',
    'nuclei segmenta', 'whole slide image', 'digital pathology',
    'oct segmenta', 'optical coherence tomograph',
]
AUTODRIVE = [
    'autonomous driv', 'self-driving', 'autonomous vehicle',
    'ego vehicle', 'nuscenes', 'waymo dataset', 'kitti dataset',
    'lane detect', 'lane segment', 'bev perception',
    "bird's-eye-view", 'bird-eye view', 'bev map',
    'traffic sign', 'drivable area', 'road detect',
    'driving perception', 'end-to-end driv', 'motion plann driv',
    'lidar camera fusion', 'sensor fusion driv',
]
REMOTESENSE = [
    'remote sensing', 'satellite imag', 'aerial imag',
    'hyperspectral', 'multispectral', 'sar image',
    'change detect remote', 'earth observation', 'overhead imag',
    'aerial scene', 'drone scene', 'uav imag',
    'geospatial', 'land cover classif', 'land use classif',
    'remote sensing classif',
]
DOCUMENT = [
    'document understand', 'document image', ' ocr ', 'text recogni',
    'scene text detect', 'scene text recogni', 'handwriting recogni',
    'table detect', 'layout analysis', 'document layout',
    'receipt understand', 'form understand', 'invoice understand',
    'chart understand', 'figure understand', 'equation recogni',
    'text line detect', 'word detect',
]
ROBOTICS = [
    'robot grasp', 'robotic manipulat', 'robot learn',
    'robot visual', 'embodied ai', 'visual navigation',
    'embodied navigat', 'robot perception', 'affordance detect',
    'manipulation policy', 'dexterous manipulat',
    'sim-to-real', 'manipulation planning',
]
SPORTS = [
    'sports analyt', 'athlete track', 'ball track',
    'game analyt', 'sport action',
]

# ---------------------------------------------------------------------------
# Reinforcement Learning
# ---------------------------------------------------------------------------

RL_KW = [
    'reinforcement learn', 'deep reinforcement', 'reward function',
    'policy gradient', 'policy optim', 'q-learning', ' dqn ', ' ppo ',
    'actor-critic', 'markov decision', ' mdp ', 'model-based rl',
    'offline rl', 'multi-agent rl', 'multi-agent reinforcement',
    'reward shaping', 'inverse reinforcement', 'imitation learn',
    'behavior clon', 'deep q-network', 'proximal policy', 'soft actor-critic',
    ' sac ', 'value function', 'temporal difference learn',
    'curiosity-driven', 'intrinsic motivat', 'exploration exploit',
    'reward maximiz', 'policy learn', 'reward signal', 'environment model',
    'mujoco', 'atari game', 'game playing', 'game agent',
]

# ---------------------------------------------------------------------------
# Optimization Theory / Learning Theory
# ---------------------------------------------------------------------------

OPTIM_THEORY = [
    'convergence rate', 'convergence analysis', 'convergence proof',
    'convergence guarantee', 'convergence theorem',
    'regret bound', 'regret minim', 'excess risk', 'sample complexity',
    'pac-bayes', 'pac learn', 'generalization bound', 'learning theory',
    'loss landscape', 'gradient flow', 'neural tangent kernel', ' ntk ',
    'double descent', 'benign overfitting', 'implicit bias',
    'overparameterized', 'optimization landscape', 'saddle point',
    'global convergence', 'linear convergence', 'online convex optim',
    'stochastic approximat', 'mirror descent', 'proximal gradient',
    'variance reduction', 'momentum method', 'acceleration method',
    'gradient norm', 'gradient clipping',
]

# ---------------------------------------------------------------------------
# Causal Inference
# ---------------------------------------------------------------------------

CAUSAL_KW = [
    'causal inference', 'causal model', ' causality', 'causal discover',
    'counterfactual', 'causal represent', 'structural causal',
    'interventional distribut', 'do-calculus', 'causal effect',
    'causal graph', 'causal mechanism',
]

# ---------------------------------------------------------------------------
# Bayesian / Probabilistic Methods
# ---------------------------------------------------------------------------

BAYES_KW = [
    'bayesian network', 'bayesian optim', 'bayesian inference',
    'gaussian process', 'variational bayes', 'bayesian neural',
    'probabilistic model', 'bayesian deep learn',
    'markov chain monte carlo', ' mcmc ', 'posterior inference',
    'stochastic variational', 'approximate inference',
    'latent dirichlet', 'dirichlet process', 'bayesian nonparametric',
    'expectation maximiz', 'em algorithm', 'variational em',
]

# ---------------------------------------------------------------------------
# Geometric Deep Learning
# ---------------------------------------------------------------------------

GEOMETRIC_KW = [
    'equivariant neural', 'geometric deep learn', 'equivariant network',
    'riemannian neural', 'manifold learn', 'geometric neural',
    'rotation equivariant', 'steerable convolution',
    'hyperbolic neural', 'hyperbolic embedding', 'hyperbolic space',
    'e(3)-equivariant', 'se(3)-equivariant', 'so(3)',
    'equivariant represent',
]

# ---------------------------------------------------------------------------
# Molecular / Scientific ML
# ---------------------------------------------------------------------------

MOLECULE_KW = [
    'molecular property', 'drug discovery', 'protein structure',
    'molecular generat', 'molecular represent', 'cheminformatics',
    'molecular fingerprint', 'drug-target', 'molecule predict',
    'binding affinity', 'protein folding', 'protein design',
    'drug design', 'molecular graph', 'chemical property',
    'molecular dynamics', 'quantum chemistry', 'material property',
]

# ---------------------------------------------------------------------------
# 6D Pose Estimation
# ---------------------------------------------------------------------------

POSE6D_KW = [
    '6d pose', '6dof pose', '6-dof pose', 'object pose estimat',
    'rigid pose', 'instance pose estimat', '6 dof',
]

# ---------------------------------------------------------------------------
# Human-Object Interaction
# ---------------------------------------------------------------------------

HOI_KW = [
    'human-object interact', 'human object interact', ' hoi ',
    'affordance detect', 'object affordance', 'human interaction detect',
]

# ---------------------------------------------------------------------------
# Multi-task / Unified Models
# ---------------------------------------------------------------------------

MULTITASK_KW = [
    'multi-task learn', 'multitask learn', 'task-agnostic',
    'auxiliary task', 'unified model for', 'universal model',
    'one model', 'one network for', 'solving multiple',
]

# ---------------------------------------------------------------------------
# NLP / Language at CVML conferences
# ---------------------------------------------------------------------------

NLP_VL_KW = [
    'language model', 'large language model', ' llm ', 'gpt-2', 'gpt-3',
    'machine translation', 'sequence-to-sequence', ' seq2seq ',
    'text summariz', 'natural language generat', 'language generation',
    'language pretrain', 'text-to-text',
    'language understanding task', 'reading comprehension',
    'named entity recogni', 'relation extraction text',
    'information extract text',
]

# ---------------------------------------------------------------------------
# Editorial / Proceedings
# ---------------------------------------------------------------------------

EDITORIAL_KW = [
    'proceedings of', 'conference on computer vision', 'international conference',
    'ieee conference', 'workshop on', 'tutorial on',
    'erratum', 'correction to', 'author index',
    'table of contents', 'foreword', 'preface',
]


# ===========================================================================
# MAIN CLASSIFIER
# ===========================================================================

def classify(title_orig: str) -> tuple[str, str, str]:
    """
    Return (Phylum, Class, Order).

    Priority order:
      1. Editorial / Proceedings
      2. Application Domains (medical, driving, remote sensing, document, robotics)
      3. 3D Vision  (NeRF, GS, depth, stereo, point cloud, SfM, scene)
      4. Generative Models (GAN, diffusion, VAE, editing, video gen)
      5. Segmentation
      6. Object Detection
      7. Human-centric Vision
      8. Vision-Language & Multimodal
      9. Video & Motion
     10. Low-level Vision
     11. Representation Learning
     12. Image Recognition & Retrieval
     13. Deep Learning Architecture
     14. Training & Learning Methods
     15. Efficient & Robust ML
     16. Fallback
    """
    t = t_low(title_orig)

    # ------------------------------------------------------------------
    # 1. Editorial / Proceedings
    # ------------------------------------------------------------------
    if _any(t, EDITORIAL_KW):
        return ('Other', 'Editorial', 'Editorial / Proceedings')

    # ------------------------------------------------------------------
    # 2. Application Domains
    # ------------------------------------------------------------------

    # 2a. Medical Imaging
    if _any(t, MEDICAL):
        # Determine class by sub-task
        if _any(t, ['segment', 'delineat', 'contour']):
            cls = 'Medical Segmentation'
            order = 'Medical Image Segmentation'
        elif _any(t, ['detect', 'recogni', 'screen', 'diagnos', 'classif']):
            cls = 'Medical Detection & Diagnosis'
            order = 'Medical Image Diagnosis'
        elif _any(t, ['super-resol', 'enhance', 'restor', 'reconstruct', 'denois', 'synthesis']):
            cls = 'Medical Image Enhancement'
            order = 'Medical Image Enhancement & Reconstruction'
        elif _any(t, ['registrat', 'align']):
            cls = 'Medical Image Registration'
            order = 'Medical Image Registration'
        else:
            cls = 'Medical Image Analysis'
            order = 'Medical Image Analysis'
        return ('14. Application Domains', cls, order)

    # 2b. Autonomous Driving
    if _any(t, AUTODRIVE):
        if _any(t, ['detect', '3d object', 'bev detect']):
            cls = 'Autonomous Driving Perception'
            order = 'AD Object Detection'
        elif _any(t, ['lane', 'road', 'drivable', 'map segment', 'bev segment']):
            cls = 'Autonomous Driving Perception'
            order = 'AD Lane & Road Segmentation'
        elif _any(t, ['track', 'forecast', 'motion predict', 'trajector']):
            cls = 'Autonomous Driving Prediction'
            order = 'AD Motion Prediction & Tracking'
        elif _any(t, ['end-to-end driv', 'plann', 'navigation']):
            cls = 'Autonomous Driving Planning'
            order = 'AD End-to-End Driving'
        elif _any(t, ['depth', 'lidar', 'sensor fusion', 'calibrat']):
            cls = 'Autonomous Driving Perception'
            order = 'AD Sensor Fusion & Depth'
        else:
            cls = 'Autonomous Driving Perception'
            order = 'AD General Perception'
        return ('14. Application Domains', cls, order)

    # 2c. Remote Sensing
    if _any(t, REMOTESENSE):
        if _any(t, ['classif', 'scene classif', 'land cover']):
            cls = 'Remote Sensing Classification'
            order = 'Remote Sensing Scene Classification'
        elif _any(t, ['detect', 'recogni']):
            cls = 'Remote Sensing Detection'
            order = 'Remote Sensing Object Detection'
        elif _any(t, ['segment', 'change detect']):
            cls = 'Remote Sensing Segmentation'
            order = 'Remote Sensing Segmentation & Change Detection'
        else:
            cls = 'Remote Sensing Analysis'
            order = 'Remote Sensing General'
        return ('14. Application Domains', cls, order)

    # 2d. Document / OCR
    if _any(t, DOCUMENT):
        if _any(t, ['scene text detect', 'text detect', 'text localiz']):
            cls = 'Document Text Detection'
            order = 'Scene Text Detection'
        elif _any(t, ['scene text recogni', 'text recogni', 'ocr ', 'handwriting']):
            cls = 'Document Text Recognition'
            order = 'Scene Text Recognition & OCR'
        elif _any(t, ['document understand', 'layout', 'table', 'form', 'receipt', 'invoice']):
            cls = 'Document Understanding'
            order = 'Document Layout & Understanding'
        else:
            cls = 'Document Analysis'
            order = 'Document Analysis General'
        return ('14. Application Domains', cls, order)

    # 2e. Robotics / Embodied AI
    if _any(t, ROBOTICS):
        if _any(t, ['grasp', 'manipulat']):
            cls = 'Robot Manipulation'
            order = 'Robotic Grasping & Manipulation'
        elif _any(t, ['navigat', 'embodied']):
            cls = 'Robot Navigation'
            order = 'Embodied Navigation'
        else:
            cls = 'Robot Perception'
            order = 'Robot Visual Perception'
        return ('14. Application Domains', cls, order)

    # ------------------------------------------------------------------
    # 3. 3D Vision & Reconstruction
    # ------------------------------------------------------------------

    # 3a. Gaussian Splatting (checked BEFORE NeRF — more specific keywords)
    if _any(t, GAUSSIAN_SPLAT):
        if _any(t, ['dynamic', '4d gaussian']):
            order = 'Dynamic Gaussian Splatting'
        elif _any(t, ['edit', 'style', 'manipulat']):
            order = 'Gaussian Splatting Editing'
        elif _any(t, ['human', 'avatar', 'body']):
            order = 'Human Gaussian Splatting'
        else:
            order = 'Gaussian Splatting'
        return ('3. 3D Vision & Reconstruction', 'Neural Implicit Representations', order)

    # 3b. NeRF / Neural Implicit
    if _any(t, NERF):
        if _any(t, ['dynamic', 'deformable', 'non-rigid']):
            order = 'Dynamic NeRF'
        elif _any(t, ['edit', 'styliz', 'manipulat']):
            order = 'NeRF Editing'
        elif _any(t, ['large-scale', 'outdoor', 'unbounded', 'city']):
            order = 'Large-scale NeRF'
        elif _any(t, ['avatar', 'human', 'body', 'face']):
            order = 'Human NeRF'
        elif _any(t, ['compress', 'efficien', 'fast', 'accelerat', 'real-time']):
            order = 'Efficient NeRF'
        else:
            order = 'Neural Radiance Fields'
        return ('3. 3D Vision & Reconstruction', 'Neural Implicit Representations', order)

    # 3c. Depth Estimation
    if _any(t, DEPTH_KW):
        if _any(t, ['self-supervised', 'unsupervised', 'self-supervis']):
            order = 'Self-supervised Depth Estimation'
        elif _any(t, ['completion', 'sparse depth']):
            order = 'Depth Completion'
        elif _any(t, ['foundation', 'zero-shot', 'generalist']):
            order = 'Foundation Model Depth'
        elif _any(t, ['video', 'temporal', 'consistent depth']):
            order = 'Video Depth Estimation'
        else:
            order = 'Monocular Depth Estimation'
        return ('3. 3D Vision & Reconstruction', 'Depth & Stereo', order)

    # 3d. Stereo
    if _any(t, STEREO_KW):
        return ('3. 3D Vision & Reconstruction', 'Depth & Stereo', 'Stereo Matching')

    # 3e. Point Cloud
    if _any(t, POINT_CLOUD):
        if _any(t, ['segment', 'semantic segment', 'part segment']):
            order = 'Point Cloud Segmentation'
        elif _any(t, ['detect', 'object detect', '3d detect']):
            order = 'Point Cloud Object Detection'
        elif _any(t, ['classif', 'recogni', 'shape classif']):
            order = 'Point Cloud Classification'
        elif _any(t, ['complet', 'reconstruct', 'upsampl']):
            order = 'Point Cloud Completion & Reconstruction'
        elif _any(t, ['registrat', 'align']):
            order = 'Point Cloud Registration'
        elif _any(t, ['flow', 'motion']):
            order = 'Point Cloud Flow Estimation'
        else:
            order = 'Point Cloud Processing'
        return ('3. 3D Vision & Reconstruction', 'Point Cloud & 3D Geometry', order)

    # 3f. SfM / SLAM / Localization
    if _any(t, SFM):
        if _any(t, ['slam', 'simultaneous localization', 'mapping']):
            order = 'SLAM & Mapping'
        elif _any(t, ['visual localiz', 'place recogni', 'relocali', 'scene coordinate']):
            order = 'Visual Localization'
        elif _any(t, ['multi-view stereo', 'mvs ', 'multi-view reconstruct']):
            order = 'Multi-view Stereo'
        elif _any(t, ['camera pose', 'pose estimat camera', 'pose regress']):
            order = 'Camera Pose Estimation'
        else:
            order = 'Structure from Motion'
        return ('3. 3D Vision & Reconstruction', 'SfM, SLAM & Localization', order)

    # 3g. Scene Reconstruction / Novel View
    if _any(t, SCENE_3D):
        if _any(t, ['novel view', 'view synthesis', 'image-based render', 'light field']):
            order = 'Novel View Synthesis'
        elif _any(t, ['surface reconstruct', 'mesh reconstruct']):
            order = 'Surface Reconstruction'
        elif _any(t, ['scene reconstruct', 'dense reconstruct', 'volumetric']):
            order = 'Scene Reconstruction'
        else:
            order = '3D Scene Understanding'
        return ('3. 3D Vision & Reconstruction', '3D Scene Understanding', order)

    # ------------------------------------------------------------------
    # 4. Generative Models & Synthesis
    # ------------------------------------------------------------------

    # 4a. Diffusion
    if _any(t, DIFFUSION):
        if _any(t, TEXTTOIMAG):
            order = 'Text-to-Image Generation'
        elif _any(t, VIDEO_GEN):
            order = 'Video Diffusion Generation'
        elif _any(t, IMG_EDIT):
            order = 'Diffusion-based Image Editing'
        elif _any(t, ['3d', 'point cloud', 'shape generat', 'mol']):
            order = '3D Diffusion Generation'
        elif _any(t, ['audio', 'speech', 'sound']):
            order = 'Audio Diffusion'
        elif _any(t, ['medical', ' mri ', 'ct scan', 'pathology']):
            order = 'Medical Diffusion'
        else:
            order = 'Diffusion Models'
        return ('6. Generative Models & Synthesis', 'Diffusion Models', order)

    # 4b. Text-to-Image (non-diffusion)
    if _any(t, TEXTTOIMAG):
        return ('6. Generative Models & Synthesis', 'Text-conditioned Generation', 'Text-to-Image Generation')

    # 4c. GAN
    if _any(t, GAN):
        if _any(t, ['face synthesis', 'face generation', 'face swap', 'face reenact', 'talking head', 'talking face']):
            order = 'Face Generation & Manipulation'
        elif _any(t, ['video synthesis', 'video generation']):
            order = 'Video Generation (GAN)'
        elif _any(t, ['image-to-image', 'pix2pix', 'cyclegan', 'style transfer']):
            order = 'Image-to-Image Translation'
        elif _any(t, ['3d', 'point cloud', 'shape']):
            order = '3D-aware GAN'
        elif _any(t, ['image editing', 'image manipulat', 'controllable']):
            order = 'GAN-based Image Editing'
        else:
            order = 'Generative Adversarial Networks'
        return ('6. Generative Models & Synthesis', 'Generative Adversarial Networks', order)

    # 4d. VAE
    if _any(t, VAE_KW):
        return ('6. Generative Models & Synthesis', 'Variational Autoencoders', 'Variational Autoencoders')

    # 4e. Image Editing (non-GAN/diffusion)
    if _any(t, IMG_EDIT):
        return ('6. Generative Models & Synthesis', 'Image Editing & Manipulation', 'Image Editing')

    # 4f. Video Generation
    if _any(t, VIDEO_GEN):
        return ('6. Generative Models & Synthesis', 'Video Generation', 'Video Generation')

    # ------------------------------------------------------------------
    # 4.5 Early Domain Adaptation check (before Segmentation)
    # Titles like "Domain Adaptation for Semantic Segmentation" should land
    # in Representation Learning, not Segmentation.
    # ------------------------------------------------------------------
    if _any(t, DOMAIN_ADAPT):
        if _any(t, ['test-time', 'tta', 'online adapt']):
            order = 'Test-time Adaptation'
        elif _any(t, ['source-free']):
            order = 'Source-free Domain Adaptation'
        elif _any(t, ['semantic segment', 'urban scene']):
            order = 'Domain Adaptive Segmentation'
        elif _any(t, ['object detect', 'domain adaptive detect']):
            order = 'Domain Adaptive Detection'
        else:
            order = 'Domain Adaptation'
        return ('7. Representation Learning', 'Domain Adaptation & Generalization', order)

    # ------------------------------------------------------------------
    # 5. Segmentation
    # ------------------------------------------------------------------

    # 5a. Video Segmentation (before generic segmentation)
    if _any(t, VID_SEG):
        if _any(t, ['semi-supervised', 'one-shot', 'zero-shot']):
            order = 'Semi-supervised Video Object Segmentation'
        elif _any(t, ['instance', 'panoptic']):
            order = 'Video Instance Segmentation'
        else:
            order = 'Video Object Segmentation'
        return ('2. Segmentation', 'Video Segmentation', order)

    # 5b. Generic / Image Segmentation
    if _any(t, SEGMENT):
        if _any(t, ['panoptic']):
            order = 'Panoptic Segmentation'
        elif _any(t, ['instance']):
            order = 'Instance Segmentation'
        elif _any(t, ['semantic', 'scene pars', 'pixel-wise']):
            order = 'Semantic Segmentation'
        elif _any(t, ['interactive', 'segment anything', ' sam ', 'click-based', 'user-guided']):
            order = 'Interactive Segmentation'
        elif _any(t, ['open-vocabulary', 'open-set', 'zero-shot', 'referring']):
            order = 'Open-Vocabulary Segmentation'
        elif _any(t, ['weakly', 'semi-supervised']):
            order = 'Weakly/Semi-supervised Segmentation'
        elif _any(t, ['3d', 'point cloud', 'lidar']):
            order = '3D Segmentation'
        else:
            order = 'Image Segmentation'
        return ('2. Segmentation', 'Image Segmentation', order)

    # ------------------------------------------------------------------
    # 6. Object Detection & Localization
    # ------------------------------------------------------------------

    # 6a. 3D Object Detection
    if _any(t, DETECT3D):
        return ('1. Object Detection & Localization', '3D Object Detection', '3D Object Detection')

    # 6b. Visual Grounding
    if _any(t, GROUNDING):
        return ('1. Object Detection & Localization', 'Visual Grounding', 'Visual Grounding & Referring')

    # 6c. Salient Object Detection
    if _any(t, SALIENT):
        return ('1. Object Detection & Localization', 'Salient Object Detection', 'Salient Object Detection')

    # 6c-extra. 6D Pose Estimation
    if _any(t, POSE6D_KW):
        return ('1. Object Detection & Localization', '6D Pose Estimation', '6D Object Pose Estimation')

    # 6d. General Object Detection
    if _any(t, DETECT):
        if _any(t, ['transformer', 'detr', 'attention-based detect']):
            order = 'Transformer-based Detection'
        elif _any(t, ['anchor-free', 'one-stage', 'fcos', 'centernet', 'atss ']):
            order = 'Anchor-free Detection'
        elif _any(t, ['open-vocabulary', 'open-set', 'zero-shot detect', 'open-world']):
            order = 'Open-Vocabulary Detection'
        elif _any(t, ['few-shot detect', 'novel category detect']):
            order = 'Few-shot Object Detection'
        elif _any(t, ['video', 'temporal detect']):
            order = 'Video Object Detection'
        elif _any(t, ['pedestrian', 'crowd', 'person detect']):
            order = 'Pedestrian Detection'
        else:
            order = 'Generic Object Detection'
        return ('1. Object Detection & Localization', '2D Object Detection', order)

    # ------------------------------------------------------------------
    # 7. Human-centric Vision
    # ------------------------------------------------------------------

    # 7a. Face (non-generation — generation handled in Generative Models)
    if _any(t, FACE):
        if _any(t, ['recogni', 'verif', 'identif', 'anti-spoof', 'liveness', 'deepfake detect']):
            order = 'Face Recognition & Verification'
        elif _any(t, ['detect', 'localiz', 'align']):
            order = 'Face Detection & Alignment'
        elif _any(t, ['age', 'attribute', 'expression']):
            order = 'Face Attribute Analysis'
        elif _any(t, ['super-resol', 'restor', 'enhance', 'hallucin']):
            order = 'Face Restoration & Enhancement'
        else:
            order = 'Face Analysis'
        return ('10. Human-centric Vision', 'Face Analysis', order)

    if _any(t, FACE_GEN):
        return ('10. Human-centric Vision', 'Face Generation', 'Face Generation & Manipulation')

    # 7b. Human Pose
    if _any(t, POSE):
        if _any(t, ['3d pose', '3d human', 'smpl', 'human mesh', 'human body model', 'mocap', 'body shape']):
            order = '3D Human Pose & Shape Estimation'
        elif _any(t, ['multi-person', 'multi person', 'crowd pose']):
            order = 'Multi-person Pose Estimation'
        elif _any(t, ['hand', 'finger']):
            order = 'Hand Pose Estimation'
        elif _any(t, ['video', 'temporal', 'motion']):
            order = 'Video Human Pose Estimation'
        else:
            order = '2D Human Pose Estimation'
        return ('10. Human-centric Vision', 'Human Pose & Body', order)

    # 7c. Person ReID
    if _any(t, REID):
        if _any(t, ['occluded', 'partial']):
            order = 'Occluded Person ReID'
        elif _any(t, ['cloth-changing', 'long-term']):
            order = 'Cloth-changing ReID'
        elif _any(t, ['vehicle', 'car reid']):
            order = 'Vehicle ReID'
        elif _any(t, ['text', 'language']):
            order = 'Text-to-Person ReID'
        else:
            order = 'Person Re-Identification'
        return ('10. Human-centric Vision', 'Person Re-Identification', order)

    # 7d. Gesture & Hand
    if _any(t, GESTURE):
        return ('10. Human-centric Vision', 'Gesture & Hand Analysis', 'Gesture & Sign Language Recognition')

    # 7e. Crowd Counting
    if _any(t, CROWD):
        return ('10. Human-centric Vision', 'Crowd Analysis', 'Crowd Counting & Density Estimation')

    # 7f. Emotion & Expression
    if _any(t, EMOTION):
        return ('10. Human-centric Vision', 'Emotion & Expression', 'Facial Expression & Emotion Recognition')

    # 7g. Gaze
    if _any(t, GAZE):
        return ('10. Human-centric Vision', 'Gaze Analysis', 'Gaze Estimation')

    # 7h. Skeleton-based Activity
    if _any(t, ACTIVITY):
        return ('10. Human-centric Vision', 'Human Activity Recognition', 'Skeleton-based Action Recognition')

    # 7i. Human-Object Interaction
    if _any(t, HOI_KW):
        return ('10. Human-centric Vision', 'Human-Object Interaction', 'Human-Object Interaction Detection')

    # ------------------------------------------------------------------
    # 8. Vision-Language & Multimodal
    # ------------------------------------------------------------------

    if _any(t, VLM):
        if _any(t, ['instruct', 'instruction follow', 'chat']):
            order = 'Visual Instruction Tuning'
        elif _any(t, ['pretrain', 'foundation', 'large-scale']):
            order = 'Vision-Language Pretraining'
        else:
            order = 'Large Vision-Language Models'
        return ('8. Vision-Language & Multimodal', 'Large Vision-Language Models', order)

    if _any(t, VQA):
        if _any(t, ['video', 'video question']):
            order = 'Video Question Answering'
        elif _any(t, ['knowledge', 'commonsense', 'reasoning']):
            order = 'Knowledge-based VQA'
        else:
            order = 'Visual Question Answering'
        return ('8. Vision-Language & Multimodal', 'Visual Question Answering', order)

    if _any(t, CAPTION):
        if _any(t, ['video', 'dense caption']):
            order = 'Dense & Video Captioning'
        else:
            order = 'Image Captioning'
        return ('8. Vision-Language & Multimodal', 'Image & Video Captioning', order)

    if _any(t, CLIP_KW):
        return ('8. Vision-Language & Multimodal', 'Contrastive Vision-Language', 'CLIP & Language-Image Pretraining')

    if _any(t, CROSSMODAL):
        if _any(t, ['audio', 'sound', 'speech']):
            order = 'Audio-Visual Learning'
        elif _any(t, ['video', 'video-text']):
            order = 'Video-Language Learning'
        elif _any(t, ['retrieval', 'image-text retrieval', 'text-image']):
            order = 'Cross-modal Retrieval'
        else:
            order = 'Multimodal Fusion'
        return ('8. Vision-Language & Multimodal', 'Cross-modal Learning', order)

    # ------------------------------------------------------------------
    # 9. Video & Motion Understanding
    # ------------------------------------------------------------------

    if _any(t, OPTFLOW):
        if _any(t, ['scene flow', '3d flow']):
            order = 'Scene Flow Estimation'
        else:
            order = 'Optical Flow Estimation'
        return ('5. Video & Motion Understanding', 'Optical Flow & Motion', order)

    if _any(t, TRACKING):
        if _any(t, ['multi-object', 'multi object track', 'multi target', 'mot ',
                    'bytetrack', 'deepsort', 'tracking-by-detect',
                    'multi pedestrian', 'multi-pedestrian',
                    'associating every', 'track every detect']):
            order = 'Multi-Object Tracking'
        elif _any(t, ['single object', 'sot ', 'visual track', 'siamese track']):
            order = 'Single Object Tracking'
        elif _any(t, ['point track', 'dense track']):
            order = 'Dense Point Tracking'
        else:
            order = 'Visual Object Tracking'
        return ('5. Video & Motion Understanding', 'Object Tracking', order)

    if _any(t, ACTION_DETECT):
        return ('5. Video & Motion Understanding', 'Temporal Action Analysis', 'Temporal Action Detection')

    if _any(t, ACTION_RECOG):
        if _any(t, ['skeleton', 'graph', 'bone', 'joint']):
            order = 'Skeleton-based Action Recognition'
        elif _any(t, ['first-person', 'egocentric', 'ego ']):
            order = 'Egocentric Action Recognition'
        elif _any(t, ['few-shot', 'zero-shot']):
            order = 'Few-shot Action Recognition'
        else:
            order = 'Video Action Recognition'
        return ('5. Video & Motion Understanding', 'Action Recognition', order)

    if _any(t, VIDEO_PREDICT):
        return ('5. Video & Motion Understanding', 'Video Prediction', 'Video Prediction & Synthesis')

    if _any(t, VIDEO_REPR):
        return ('5. Video & Motion Understanding', 'Video Representation Learning', 'Video Self-supervised Learning')

    if _any(t, TRAJECTORY):
        return ('5. Video & Motion Understanding', 'Trajectory Prediction', 'Trajectory Prediction & Forecasting')

    # Generic video
    if _any(t, ['video ', ' video', 'temporal ']):
        if _any(t, ['understand', 'analyz', 'retrieval', 'search', 'localiz']):
            return ('5. Video & Motion Understanding', 'Video Understanding', 'Video Understanding')
        elif _any(t, ['object', 'detect', 'track', 'segment']):
            return ('5. Video & Motion Understanding', 'Video Object Analysis', 'Video Object Analysis')
        elif _any(t, ['represent', 'pretrain', 'self-supervised']):
            return ('5. Video & Motion Understanding', 'Video Representation Learning', 'Video Representation Learning')

    # ------------------------------------------------------------------
    # 10. Low-level Vision
    # ------------------------------------------------------------------

    if _any(t, SUPERRES):
        if _any(t, ['blind', 'real-world', 'unknown degradat']):
            order = 'Blind Super-Resolution'
        elif _any(t, ['video']):
            order = 'Video Super-Resolution'
        elif _any(t, ['face']):
            order = 'Face Super-Resolution'
        else:
            order = 'Single Image Super-Resolution'
        return ('9. Low-level Vision', 'Image Super-Resolution', order)

    if _any(t, RESTORE):
        if _any(t, ['denois', 'noise remov', 'image noise']):
            order = 'Image Denoising'
        elif _any(t, ['deblur', 'motion blur', 'defocus']):
            order = 'Image Deblurring'
        elif _any(t, ['derain', 'rain']):
            order = 'Image Deraining'
        elif _any(t, ['dehaz', 'fog', 'haze']):
            order = 'Image Dehazing'
        elif _any(t, ['artifact', 'jpeg', 'compression artifact']):
            order = 'Image Artifact Removal'
        elif _any(t, ['all-in-one', 'unified restor', 'adverse weather']):
            order = 'All-in-one Image Restoration'
        else:
            order = 'Image Restoration'
        return ('9. Low-level Vision', 'Image Restoration', order)

    if _any(t, ENHANCE):
        if _any(t, ['low-light', 'illumination', 'retinex']):
            order = 'Low-light Image Enhancement'
        elif _any(t, ['hdr', 'tone map']):
            order = 'HDR Imaging'
        elif _any(t, ['color', 'colour']):
            order = 'Image Colorization & Color Enhancement'
        else:
            order = 'Image Enhancement'
        return ('9. Low-level Vision', 'Image Enhancement', order)

    if _any(t, IMG_COMPRESS):
        return ('9. Low-level Vision', 'Image & Video Compression', 'Learned Image/Video Compression')

    if _any(t, IMG_QUALITY):
        return ('9. Low-level Vision', 'Image Quality Assessment', 'No-reference Image Quality Assessment')

    if _any(t, COMP_PHOTO):
        return ('9. Low-level Vision', 'Computational Photography', 'Computational Photography')

    # ------------------------------------------------------------------
    # 11. Representation Learning
    # ------------------------------------------------------------------

    if _any(t, MAE):
        if _any(t, ['video', 'temporal']):
            order = 'Video Masked Autoencoding'
        else:
            order = 'Masked Autoencoders (MAE)'
        return ('7. Representation Learning', 'Masked Image Modeling', order)

    if _any(t, SSL):
        if _any(t, ['video', 'temporal']):
            order = 'Video Self-supervised Learning'
        elif _any(t, ['medical', 'pathology', 'clinical']):
            order = 'Medical Self-supervised Learning'
        else:
            order = 'Contrastive Self-supervised Learning'
        return ('7. Representation Learning', 'Self-supervised Learning', order)

    if _any(t, DOMAIN_ADAPT):
        if _any(t, ['test-time', 'tta', 'online adapt']):
            order = 'Test-time Adaptation'
        elif _any(t, ['source-free']):
            order = 'Source-free Domain Adaptation'
        elif _any(t, ['semantic segment', 'urban scene']):
            order = 'Domain Adaptive Segmentation'
        elif _any(t, ['object detect', 'domain adaptive detect']):
            order = 'Domain Adaptive Detection'
        else:
            order = 'Domain Adaptation'
        return ('7. Representation Learning', 'Domain Adaptation & Generalization', order)

    if _any(t, TRANSFER):
        if _any(t, ['prompt tuning', 'visual prompt', 'prefix tuning', 'adapter']):
            order = 'Parameter-efficient Fine-tuning'
        elif _any(t, ['fine-tuning', 'fine-tune']):
            order = 'Fine-tuning Pretrained Models'
        else:
            order = 'Transfer Learning'
        return ('7. Representation Learning', 'Transfer Learning', order)

    if _any(t, FEW_SHOT_REP):
        if _any(t, ['zero-shot', 'zero shot']):
            order = 'Zero-shot Learning'
        elif _any(t, ['meta-learn', 'meta learn', 'learning to learn']):
            order = 'Meta-learning'
        else:
            order = 'Few-shot Learning'
        return ('7. Representation Learning', 'Few-shot & Meta-learning', order)

    if _any(t, KG_EMBED):
        if _any(t, ['scene graph', 'visual relation', 'visual relationship']):
            order = 'Scene Graph Generation'
        else:
            order = 'Knowledge Graph Embedding'
        return ('7. Representation Learning', 'Knowledge & Relational Learning', order)

    if _any(t, SPARSE_REPR):
        if _any(t, ['pca ', 'principal component', 'dimensionality']):
            order = 'Dimensionality Reduction'
        elif _any(t, ['compressed sensing', 'compressive sensing']):
            order = 'Compressed Sensing'
        else:
            order = 'Sparse Coding & Dictionary Learning'
        return ('7. Representation Learning', 'Sparse & Dictionary Representations', order)

    if _any(t, CLUSTERING_KW):
        return ('7. Representation Learning', 'Unsupervised Clustering', 'Deep Clustering')

    # ------------------------------------------------------------------
    # 12. Deep Learning Architecture (before Image Recognition to avoid
    #     ViT/architecture papers being misclassified as recognition)
    # ------------------------------------------------------------------

    if _any(t, STATE_SPACE):
        return ('11. Deep Learning Architecture', 'State Space Models', 'Vision State Space Models (Mamba)')

    if _any(t, VIT_ARCH):
        if _any(t, ['swin', 'pyramid', 'hierarchical']):
            order = 'Hierarchical Vision Transformers'
        elif _any(t, ['efficient', 'lightweight', 'mobile vit']):
            order = 'Efficient Vision Transformers'
        else:
            order = 'Vision Transformers (ViT)'
        return ('11. Deep Learning Architecture', 'Vision Transformers', order)

    if _any(t, GNN_ARCH):
        if _any(t, ['spatial-temporal', 'st-gcn', 'dynamic graph']):
            order = 'Spatial-Temporal Graph Networks'
        elif _any(t, ['heterogeneous', 'knowledge graph']):
            order = 'Heterogeneous Graph Networks'
        else:
            order = 'Graph Neural Networks'
        return ('11. Deep Learning Architecture', 'Graph Neural Networks', order)

    if _any(t, NAS):
        return ('11. Deep Learning Architecture', 'Neural Architecture Search', 'Neural Architecture Search')

    if _any(t, ATTENTION):
        return ('11. Deep Learning Architecture', 'Attention Mechanisms', 'Attention Mechanisms in Deep Learning')

    if _any(t, CNN_ARCH):
        if _any(t, ['residual', 'resnet', 'skip connect']):
            order = 'Residual & Dense Networks'
        elif _any(t, ['mobile', 'lightweight', 'efficientnet', 'shufflenet']):
            order = 'Lightweight CNN Architectures'
        elif _any(t, ['deformable', 'dynamic']):
            order = 'Deformable & Dynamic Convolutions'
        else:
            order = 'Convolutional Neural Network Architectures'
        return ('11. Deep Learning Architecture', 'Convolutional Networks', order)

    if _any(t, MLP_ARCH):
        return ('11. Deep Learning Architecture', 'MLP-based Architectures', 'MLP-Mixer & Vision MLP')

    if _any(t, TRANSFORMER_GEN):
        return ('11. Deep Learning Architecture', 'Transformer Architectures', 'Transformer Architectures for Vision')

    # ------------------------------------------------------------------
    # 13. Image Recognition & Retrieval
    # ------------------------------------------------------------------

    if _any(t, FEAT_MATCH):
        return ('4. Image Recognition & Retrieval', 'Feature Matching', 'Local Feature Matching')

    if _any(t, RETRIEVAL):
        if _any(t, ['hash', 'compact', 'binary code']):
            order = 'Hashing-based Retrieval'
        elif _any(t, ['cross-modal', 'image-text', 'text-image']):
            order = 'Cross-modal Retrieval'
        elif _any(t, ['sketch', 'sbir']):
            order = 'Sketch-based Image Retrieval'
        else:
            order = 'Image Retrieval'
        return ('4. Image Recognition & Retrieval', 'Image Retrieval', order)

    if _any(t, METRIC):
        return ('4. Image Recognition & Retrieval', 'Metric Learning', 'Deep Metric Learning')

    if _any(t, FINEGRAINED):
        return ('4. Image Recognition & Retrieval', 'Fine-grained Recognition', 'Fine-grained Visual Categorization')

    if _any(t, SCENE_RECOG):
        return ('4. Image Recognition & Retrieval', 'Scene Recognition', 'Scene & Place Recognition')

    if _any(t, CLASSIF):
        if _any(t, ['multi-label', 'multi-class']):
            order = 'Multi-label Classification'
        elif _any(t, ['medical', 'clinical', 'pathology']):
            order = 'Medical Image Classification'
        elif _any(t, ['long-tail', 'imbalanced', 'class imbalance']):
            order = 'Long-tail Classification'
        else:
            order = 'Image Classification'
        return ('4. Image Recognition & Retrieval', 'Image Classification', order)

    # ------------------------------------------------------------------
    # 14. Training & Learning Methods
    # ------------------------------------------------------------------

    if _any(t, KD):
        if _any(t, ['feature', 'intermediate']):
            order = 'Feature-based Knowledge Distillation'
        elif _any(t, ['online', 'mutual learn', 'born again']):
            order = 'Online Knowledge Distillation'
        else:
            order = 'Knowledge Distillation'
        return ('12. Training & Learning Methods', 'Knowledge Distillation', order)

    if _any(t, CONTINUAL):
        if _any(t, ['class-incremental', 'task-incremental']):
            order = 'Class/Task-incremental Learning'
        elif _any(t, ['exemplar', 'replay', 'memory']):
            order = 'Replay-based Continual Learning'
        else:
            order = 'Continual & Incremental Learning'
        return ('12. Training & Learning Methods', 'Continual Learning', order)

    if _any(t, SEMISUP):
        if _any(t, ['segment', 'dense predict']):
            order = 'Semi-supervised Segmentation'
        elif _any(t, ['detect', 'object detect']):
            order = 'Semi-supervised Detection'
        elif _any(t, ['classif', 'recogni']):
            order = 'Semi-supervised Classification'
        else:
            order = 'Semi-supervised Learning'
        return ('12. Training & Learning Methods', 'Semi-supervised Learning', order)

    if _any(t, WEAKLY):
        if _any(t, ['segment', 'cam']):
            order = 'Weakly Supervised Segmentation'
        elif _any(t, ['detect', 'localiz']):
            order = 'Weakly Supervised Detection'
        else:
            order = 'Weakly Supervised Learning'
        return ('12. Training & Learning Methods', 'Weakly Supervised Learning', order)

    if _any(t, AUGMENT):
        return ('12. Training & Learning Methods', 'Data Augmentation', 'Data Augmentation Strategies')

    if _any(t, LONGTAIL):
        return ('12. Training & Learning Methods', 'Long-tail & Imbalanced Learning', 'Long-tail Learning')

    if _any(t, NOISY_LBL):
        return ('12. Training & Learning Methods', 'Noisy Label Learning', 'Learning with Noisy Labels')

    if _any(t, CURRICULUM):
        return ('12. Training & Learning Methods', 'Curriculum Learning', 'Curriculum & Self-paced Learning')

    if _any(t, ACTIVE):
        return ('12. Training & Learning Methods', 'Active Learning', 'Active Learning Strategies')

    if _any(t, OPTIM):
        return ('12. Training & Learning Methods', 'Optimization Methods', 'Optimization & Training Dynamics')

    # 14-rl. Reinforcement Learning
    if _any(t, RL_KW):
        if _any(t, ['imitation learn', 'behavior clon', 'inverse rl', 'inverse reinforcement']):
            order = 'Imitation & Inverse RL'
        elif _any(t, ['multi-agent', 'cooperative', 'competitive rl']):
            order = 'Multi-agent RL'
        elif _any(t, ['offline rl', 'offline reinforcement', 'batch rl']):
            order = 'Offline Reinforcement Learning'
        elif _any(t, ['model-based']):
            order = 'Model-based RL'
        elif _any(t, ['visual', 'vision', 'image', 'pixel', 'camera']):
            order = 'Visual Reinforcement Learning'
        else:
            order = 'Deep Reinforcement Learning'
        return ('12. Training & Learning Methods', 'Reinforcement Learning', order)

    # 14-theory. Optimization Theory
    if _any(t, OPTIM_THEORY):
        return ('12. Training & Learning Methods', 'Optimization Theory', 'Optimization Theory & Convergence')

    # 14-multi. Multi-task Learning
    if _any(t, MULTITASK_KW):
        return ('12. Training & Learning Methods', 'Multi-task Learning', 'Multi-task & Joint Learning')

    # ------------------------------------------------------------------
    # 15. Efficient & Robust ML
    # ------------------------------------------------------------------

    if _any(t, ADV):
        if _any(t, ['physical', 'patch', 'universal']):
            order = 'Physical & Universal Adversarial Attacks'
        elif _any(t, ['transfer', 'black-box', 'query-based']):
            order = 'Transferable Adversarial Attacks'
        else:
            order = 'Adversarial Attacks'
        return ('13. Efficient & Robust ML', 'Adversarial Robustness', order)

    if _any(t, ADV_DEF):
        return ('13. Efficient & Robust ML', 'Adversarial Robustness', 'Adversarial Defense & Certified Robustness')

    if _any(t, OOD):
        if _any(t, ['anomaly', 'novelty', 'outlier']):
            order = 'Anomaly & Novelty Detection'
        elif _any(t, ['open-set']):
            order = 'Open-set Recognition'
        else:
            order = 'Out-of-distribution Detection'
        return ('13. Efficient & Robust ML', 'OOD & Anomaly Detection', order)

    if _any(t, PRUNING):
        if _any(t, ['lottery ticket', 'sparse']):
            order = 'Sparse & Lottery Ticket Networks'
        else:
            order = 'Network Pruning'
        return ('13. Efficient & Robust ML', 'Model Compression', 'Network Pruning')

    if _any(t, QUANT):
        return ('13. Efficient & Robust ML', 'Model Compression', 'Network Quantization')

    if _any(t, COMPRESS_ML):
        return ('13. Efficient & Robust ML', 'Model Compression', 'Model Compression & Efficient Inference')

    if _any(t, UNCERTAINTY):
        return ('13. Efficient & Robust ML', 'Uncertainty & Calibration', 'Uncertainty Estimation & Calibration')

    if _any(t, XAI):
        return ('13. Efficient & Robust ML', 'Explainability & Interpretability', 'Visual Explanations & Attribution')

    if _any(t, FAIRNESS):
        return ('13. Efficient & Robust ML', 'Fairness & Bias', 'Fairness, Bias & Debiasing')

    if _any(t, FEDERATED):
        return ('13. Efficient & Robust ML', 'Federated & Distributed Learning', 'Federated Learning')

    if _any(t, PRIVACY):
        return ('13. Efficient & Robust ML', 'Privacy & Security', 'Differential Privacy & Machine Unlearning')

    # 15-ext. Causal Inference
    if _any(t, CAUSAL_KW):
        return ('13. Efficient & Robust ML', 'Causal Inference', 'Causal Inference & Counterfactuals')

    # 15-ext. Bayesian / Probabilistic Methods
    if _any(t, BAYES_KW):
        if _any(t, ['gaussian process']):
            order = 'Gaussian Processes'
        elif _any(t, ['optim', 'hyperparameter']):
            order = 'Bayesian Optimization'
        else:
            order = 'Bayesian Deep Learning'
        return ('13. Efficient & Robust ML', 'Bayesian & Probabilistic Methods', order)

    # 15-ext. Geometric Deep Learning
    if _any(t, GEOMETRIC_KW):
        return ('11. Deep Learning Architecture', 'Geometric Deep Learning', 'Equivariant & Geometric Neural Networks')

    # 16-app. Molecular / Scientific ML
    if _any(t, MOLECULE_KW):
        return ('14. Application Domains', 'Scientific & Molecular ML', 'Molecular Property Prediction & Drug Discovery')

    # 16-nlp. Language Models / NLP
    if _any(t, NLP_VL_KW):
        return ('8. Vision-Language & Multimodal', 'Language Model Applications', 'Language Models & NLP')

    # ------------------------------------------------------------------
    # 16. Aggressive generic fallbacks (order matters — specific first)
    # ------------------------------------------------------------------

    # Segmentation (any remaining)
    if _any(t, ['segmentation', ' segment ', 'scene parsing', 'pixel-wise']):
        return ('2. Segmentation', 'Image Segmentation', 'Image Segmentation')

    # Detection (any remaining)
    if _any(t, ['detection ', 'detector ', ' localization', 'bounding box']):
        return ('1. Object Detection & Localization', '2D Object Detection', 'Generic Object Detection')

    # Tracking
    if _any(t, ['tracking ', ' tracking', 'visual tracker']):
        return ('5. Video & Motion Understanding', 'Object Tracking', 'Visual Object Tracking')

    # Generation / synthesis
    if _any(t, [' generation', 'generative model', ' synthesis ', 'image synthesis']):
        return ('6. Generative Models & Synthesis', 'Generative Models', 'Generative Models')

    # 3D remaining
    if _any(t, [' 3d ', '3-d ', 'three-dimensional', 'depth ', 'voxel ']):
        return ('3. 3D Vision & Reconstruction', '3D Scene Understanding', '3D Scene Understanding')

    # Video remaining
    if _any(t, ['video ', ' video', 'temporal ']):
        return ('5. Video & Motion Understanding', 'Video Understanding', 'Video Understanding')

    # Classification / recognition
    if _any(t, ['classification', 'classifier', 'categorization', ' recognition']):
        return ('4. Image Recognition & Retrieval', 'Image Classification', 'Image Classification')

    # Representation / embedding
    if _any(t, ['representation learn', 'feature learn', 'embedding learn', 'visual feature']):
        return ('7. Representation Learning', 'Representation Learning', 'General Representation Learning')

    # Multi-modal remaining
    if _any(t, ['multimodal', 'multi-modal', 'vision and language', 'language and vision']):
        return ('8. Vision-Language & Multimodal', 'Multimodal Learning', 'Multimodal Learning')

    # Training / learning methods remaining
    if _any(t, ['regulariz', 'batch normaliz', 'layer normaliz', 'dropout ',
                'data-efficient', 'label-efficient', 'efficient train']):
        return ('12. Training & Learning Methods', 'Training Techniques', 'Regularization & Training Techniques')

    # Efficient inference remaining
    if _any(t, ['inference ', 'efficient network', 'compact model', 'real-time ']):
        return ('13. Efficient & Robust ML', 'Model Compression', 'Model Compression & Efficient Inference')

    # General neural network / deep learning
    if _any(t, ['deep learning', 'deep neural', 'neural network', 'convolutional network',
                'deep model', 'deep architecture']):
        return ('11. Deep Learning Architecture', 'General Deep Learning', 'Deep Neural Networks')

    # Image remaining
    if _any(t, ['image ', ' image', 'visual ', 'vision ', 'photo ', 'pixel']):
        return ('4. Image Recognition & Retrieval', 'Visual Analysis', 'General Visual Analysis')

    # Learning remaining
    if _any(t, [' learning', 'machine learning', 'learn from']):
        return ('12. Training & Learning Methods', 'Training Techniques', 'General Learning Methods')

    # ------------------------------------------------------------------
    # 17. Final fallback
    # ------------------------------------------------------------------
    return ('Other', 'Unclassified', 'Unclassified')


# ===========================================================================
# Quick smoke test (run directly)
# ===========================================================================

if __name__ == '__main__':
    test_cases = [
        ("Deep Residual Learning for Image Recognition",
         ('4. Image Recognition & Retrieval', 'Image Classification', 'Image Classification')),
        ("Attention Is All You Need",
         ('11. Deep Learning Architecture', 'Attention Mechanisms', 'Attention Mechanisms in Deep Learning')),
        ("NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
         ('3. 3D Vision & Reconstruction', 'Neural Implicit Representations', 'Neural Radiance Fields')),
        ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
         ('11. Deep Learning Architecture', 'Vision Transformers', 'Vision Transformers (ViT)')),
        ("Masked Autoencoders Are Scalable Vision Learners",
         ('7. Representation Learning', 'Masked Image Modeling', 'Masked Autoencoders (MAE)')),
        ("Segment Anything",
         ('2. Segmentation', 'Image Segmentation', 'Interactive Segmentation')),
        ("DETR: End-to-End Object Detection with Transformers",
         ('1. Object Detection & Localization', '2D Object Detection', 'Transformer-based Detection')),
        ("3D Gaussian Splatting for Real-Time Radiance Field Rendering",
         ('3. 3D Vision & Reconstruction', 'Neural Implicit Representations', 'Gaussian Splatting')),
        ("Denoising Diffusion Probabilistic Models",
         ('6. Generative Models & Synthesis', 'Diffusion Models', 'Diffusion Models')),
        ("Medical Image Segmentation with U-Net",
         ('14. Application Domains', 'Medical Segmentation', 'Medical Image Segmentation')),
        ("Autonomous Driving with BEV Perception",
         ('14. Application Domains', 'Autonomous Driving Perception', 'AD General Perception')),
        ("Scene Text Recognition with Deep Learning",
         ('14. Application Domains', 'Document Text Recognition', 'Scene Text Recognition & OCR')),
        ("Human Pose Estimation via Deep Learning",
         ('10. Human-centric Vision', 'Human Pose & Body', '2D Human Pose Estimation')),
        ("Face Recognition via ArcFace",
         ('10. Human-centric Vision', 'Face Analysis', 'Face Recognition & Verification')),
        ("Optical Flow Estimation with PWC-Net",
         ('5. Video & Motion Understanding', 'Optical Flow & Motion', 'Optical Flow Estimation')),
        ("Domain Adaptation for Semantic Segmentation",
         ('7. Representation Learning', 'Domain Adaptation & Generalization', 'Domain Adaptive Segmentation')),
        ("Point Cloud Classification via PointNet",
         ('3. 3D Vision & Reconstruction', 'Point Cloud & 3D Geometry', 'Point Cloud Classification')),
        ("Knowledge Distillation for Efficient Neural Networks",
         ('12. Training & Learning Methods', 'Knowledge Distillation', 'Knowledge Distillation')),
        ("Adversarial Examples in the Physical World",
         ('13. Efficient & Robust ML', 'Adversarial Robustness', 'Physical & Universal Adversarial Attacks')),
        ("CLIP: Learning Transferable Visual Models from Natural Language Supervision",
         ('8. Vision-Language & Multimodal', 'Contrastive Vision-Language', 'CLIP & Language-Image Pretraining')),
    ]

    passed = 0
    failed = 0
    print("=" * 70)
    print("CVML Classifier Smoke Test")
    print("=" * 70)
    for title, expected in test_cases:
        result = classify(title)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {title[:55]:<55}")
        if not ok:
            print(f"       Expected : {expected}")
            print(f"       Got      : {result}")
    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")
    print("=" * 70)

    # Batch test on a small slice of real data
    import json, os
    data_path = os.path.join(os.path.dirname(__file__), 'all_dblp.json')
    if os.path.exists(data_path):
        with open(data_path, encoding='utf-8') as f:
            papers = json.load(f)
        sample = papers[:500]
        other_count = 0
        phylum_counts: dict[str, int] = {}
        for p in sample:
            ph, cl, od = classify(p['title'])
            phylum_counts[ph] = phylum_counts.get(ph, 0) + 1
            if ph == 'Other':
                other_count += 1
        print(f"\nSample batch (first 500 papers):")
        for ph, cnt in sorted(phylum_counts.items(), key=lambda x: -x[1]):
            print(f"  {cnt:4d}  {ph}")
        print(f"\nOther/Unclassified rate: {other_count/len(sample)*100:.1f}%")
