"""
Video DeepDream

This tool is heavily based on "The Annotated Deep Dream" project which has been slightly adapted for real-time use. 
Credits for most of the code go to Gordic Aleksa
"""

"""
Imports
"""

import sys
import os

import enum
from collections import namedtuple
import argparse
import numbers
import math
import warnings
import threading

# Deep learning related imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

# Rendering Inputs
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# GUI imports
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# OSC imports
from pythonosc import dispatcher
from pythonosc import osc_server

# Ignore harmless PyTorch and Torchvision deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Force PyTorch to download and look for models in a local 'models' folder
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'models')

"""
Video Settings
"""
image_resolution = (1280, 720)
use_live_camera_input = True
movie_file_path = "../../../Data/Video/Stocos/Solos/Take4_Blumen_Baile.mp4"

"""
Supported Models
"""
class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1

class SupportedModels(enum.Enum):
    VGG16_EXPERIMENTAL = 0
    RESNET50 = 1
    MOBILENET_V2 = 2
    ALEXNET = 3

"""
Deep Dream Configuration
"""

"""
# VGG16_EXPERIMENTAL
config = {
    'dump_dir': "results/images",
    "input": "",
    "img_width": image_resolution[0],
    "layers_to_use": ["conv4_2"],
    "features_to_use": [226],
    "use_noise": False,
    "pyramid_size": 2,
    "pyramid_ratio": 1.1,
    "num_gradient_ascent_iterations": 1,
    "lr": 0.09,
    "should_display": False,
    "spatial_shift_size": 0,
    "smoothing_coefficient": 0.5,
    "image_blend_factor": 0.1,
    "use_bg_subtraction": False,
    "model_name": SupportedModels.VGG16_EXPERIMENTAL.name,
    "pretrained_weights": SupportedPretrainedWeights.IMAGENET.name
}
"""

# MobileNetV2
config = {
    'dump_dir': "results/images",
    "input": "",
    "img_width": image_resolution[0],
    "layers_to_use": ["block_14"],
    "features_to_use": [0],
    "use_noise": False,
    "pyramid_size": 2,
    "pyramid_ratio": 1.1,
    "num_gradient_ascent_iterations": 1,
    "lr": 0.09,
    "should_display": False,
    "spatial_shift_size": 0,
    "smoothing_coefficient": 0.5,
    "image_blend_factor": 0.1,
    "use_bg_subtraction": False,
    "model_name": SupportedModels.MOBILENET_V2.name,
    "pretrained_weights": SupportedPretrainedWeights.IMAGENET.name
}

"""
# AlexNet
config = {
    'dump_dir': "results/images",
    "input": "",
    "img_width": image_resolution[0],
    "layers_to_use": ["layer_8"],
    "features_to_use": [0],
    "use_noise": False,
    "pyramid_size": 2,
    "pyramid_ratio": 1.1,
    "num_gradient_ascent_iterations": 1,
    "lr": 0.09,
    "should_display": False,
    "spatial_shift_size": 0,
    "smoothing_coefficient": 0.5,
    "image_blend_factor": 0.1,
    "use_bg_subtraction": False,
    "model_name": SupportedModels.ALEXNET.name,
    "pretrained_weights": SupportedPretrainedWeights.IMAGENET.name
}
"""

"""
OSC Settings
"""
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9004

"""
Compute Device
"""
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(DEVICE))

if DEVICE.type == 'cuda':
    cudnn.benchmark = True

"""
Other Stuff
"""
    
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, 'input')
BINARIES_PATH = os.path.join(os.getcwd(), 'models', 'binaries')
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, 'out-images')

os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

"""
Exposing neural network's activations
"""
class Vgg16Experimental(torch.nn.Module):
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            vgg16 = models.vgg16(pretrained=True, progress=show_progress).eval()
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported.')

        vgg_pretrained_features = vgg16.features
        self.conv1_1 = vgg_pretrained_features[0]
        self.relu1_1 = vgg_pretrained_features[1]
        self.conv1_2 = vgg_pretrained_features[2]
        self.relu1_2 = vgg_pretrained_features[3]
        self.max_pooling1 = vgg_pretrained_features[4]
        self.conv2_1 = vgg_pretrained_features[5]
        self.relu2_1 = vgg_pretrained_features[6]
        self.conv2_2 = vgg_pretrained_features[7]
        self.relu2_2 = vgg_pretrained_features[8]
        self.max_pooling2 = vgg_pretrained_features[9]
        self.conv3_1 = vgg_pretrained_features[10]
        self.relu3_1 = vgg_pretrained_features[11]
        self.conv3_2 = vgg_pretrained_features[12]
        self.relu3_2 = vgg_pretrained_features[13]
        self.conv3_3 = vgg_pretrained_features[14]
        self.relu3_3 = vgg_pretrained_features[15]
        self.max_pooling3 = vgg_pretrained_features[16]
        self.conv4_1 = vgg_pretrained_features[17]
        self.relu4_1 = vgg_pretrained_features[18]
        self.conv4_2 = vgg_pretrained_features[19]
        self.relu4_2 = vgg_pretrained_features[20]
        self.conv4_3 = vgg_pretrained_features[21]
        self.relu4_3 = vgg_pretrained_features[22]
        self.max_pooling4 = vgg_pretrained_features[23]
        self.conv5_1 = vgg_pretrained_features[24]
        self.relu5_1 = vgg_pretrained_features[25]
        self.conv5_2 = vgg_pretrained_features[26]
        self.relu5_2 = vgg_pretrained_features[27]
        self.conv5_3 = vgg_pretrained_features[28]
        self.relu5_3 = vgg_pretrained_features[29]
        self.max_pooling5 = vgg_pretrained_features[30]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        self.layer_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2",
                            "conv3_1", "conv3_2", "conv3_3", "conv4_1",
                            "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
        self.layer_outputs = {layer_name: None for layer_name in self.layer_names}

    def forward(self, x):
        x = self.conv1_1(x)
        self.layer_outputs["conv1_1"] = x
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        self.layer_outputs["conv1_2"] = x
        x = self.relu1_2(x)
        x = self.max_pooling1(x)
        
        x = self.conv2_1(x)
        self.layer_outputs["conv2_1"] = x
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        self.layer_outputs["conv2_2"] = x
        x = self.relu2_2(x)
        x = self.max_pooling2(x)
        
        x = self.conv3_1(x)
        self.layer_outputs["conv3_1"] = x
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        self.layer_outputs["conv3_2"] = x
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        self.layer_outputs["conv3_3"] = x
        x = self.relu3_3(x)
        x = self.max_pooling3(x)
        
        x = self.conv4_1(x)
        self.layer_outputs["conv4_1"] = x
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        self.layer_outputs["conv4_2"] = x
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        self.layer_outputs["conv4_3"] = x
        x = self.relu4_3(x)
        x = self.max_pooling4(x)
        
        x = self.conv5_1(x)
        self.layer_outputs["conv5_1"] = x
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        self.layer_outputs["conv5_2"] = x
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        self.layer_outputs["conv5_3"] = x
        x = self.relu5_3(x)
        
        return x, self.layer_outputs

class MobileNetV2Experimental(torch.nn.Module):
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True, progress=show_progress).eval()
        self.features = mobilenet.features
        self.layer_names = [f"block_{i}" for i in range(len(self.features))]
        self.layer_outputs = {layer_name: None for layer_name in self.layer_names}

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i, block in enumerate(self.features):
            x = block(x)
            self.layer_outputs[f"block_{i}"] = x
        return x, self.layer_outputs

class AlexNetExperimental(torch.nn.Module):
    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        alexnet = models.alexnet(pretrained=True, progress=show_progress).eval()
        self.features = alexnet.features
        self.layer_names = [f"layer_{i}" for i in range(len(self.features))]
        self.layer_outputs = {layer_name: None for layer_name in self.layer_names}

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            self.layer_outputs[f"layer_{i}"] = x
        return x, self.layer_outputs
    
def fetch_and_prepare_model(model_type, pretrained_weights):
    if model_type == SupportedModels.VGG16_EXPERIMENTAL.name:
        model = Vgg16Experimental(pretrained_weights, requires_grad=False, show_progress=True).to(DEVICE)
    elif model_type == SupportedModels.MOBILENET_V2.name:
        model = MobileNetV2Experimental(pretrained_weights, requires_grad=False, show_progress=True).to(DEVICE)
    elif model_type == SupportedModels.ALEXNET.name:
        model = AlexNetExperimental(pretrained_weights, requires_grad=False, show_progress=True).to(DEVICE)
    else:
        raise Exception('Model not yet supported.')
    
    if DEVICE.type == 'cuda':
        model = model.half()
        
    return model

"""
Image loading, saving and displaying
"""
def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)
    img /= 255.0
    return img

"""
DeepDream image/tensor utilities
"""
def pre_process_numpy_img(img):
    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1
    return img

def post_process_numpy_img(img):
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
    mean = IMAGENET_MEAN_1.reshape(1, 1, -1)
    std = IMAGENET_STD_1.reshape(1, 1, -1)
    img = (img * std) + mean
    img = np.clip(img, 0., 1.)
    return img

def pytorch_input_adapter(img):
    tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    if DEVICE.type == 'cuda':
        tensor = tensor.half()
    tensor.requires_grad = True
    return tensor

def pytorch_output_adapter(tensor):
    return np.moveaxis(tensor.to('cpu').detach().float().numpy()[0], 0, 2)

def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        if DEVICE.type == 'cuda':
            rolled = rolled.half()
        rolled.requires_grad = True
        return rolled

"""
Image pyramid
"""
def get_new_shape(pyramid_ratio, pyramid_size, original_shape, pyramid_level):
    scale = (1.0 / pyramid_ratio) ** (pyramid_size - pyramid_level - 1)
    return int(round(original_shape[0] * scale)), int(round(original_shape[1] * scale))

def deep_dream_static_image(config, img=None):
    try:
        layers_to_use = [layer_name for layer_name in config['layers_to_use']]
        features_to_use = [feature_index for feature_index in config['features_to_use']]
    except Exception as e:
        print(f'Invalid layer names. Available: {model.layer_names}.')
        return

    if img is None:
        img_path = os.path.join(INPUT_DATA_PATH, config['input'])
        img = load_image(img_path, target_shape=config['img_width'])
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = pre_process_numpy_img(img)
    original_shape = img.shape[:-1]

    pyramid_ratio = config['pyramid_ratio']
    pyramid_size = config['pyramid_size']

    for pyramid_level in range(pyramid_size):
        new_shape = get_new_shape(pyramid_ratio, pyramid_size, original_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = pytorch_input_adapter(img)

        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = random_circular_spatial_shift(input_tensor, h_shift, w_shift)

            gradient_ascent(config, model, input_tensor, layers_to_use, features_to_use, iteration)

            input_tensor = random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)

        img = pytorch_output_adapter(input_tensor)

    return post_process_numpy_img(img)

"""
Gradient Ascent Core
"""
class CascadeGaussianSmoothing(nn.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]

        self.pad = int(kernel_size[0] / 2)

        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            kernel = kernel / torch.sum(kernel)
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1).to(DEVICE)
            if DEVICE.type == 'cuda':
                kernel = kernel.half()
            gaussian_kernels.append(kernel)

        self.weight1 = nn.Parameter(gaussian_kernels[0], requires_grad=False)
        self.weight2 = nn.Parameter(gaussian_kernels[1], requires_grad=False)
        self.weight3 = nn.Parameter(gaussian_kernels[2], requires_grad=False)
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)
        return (grad1 + grad2 + grad3) / 3

def gradient_ascent(config, model, input_tensor, layers_to_use, features_to_use, iteration):
    _, out = model(input_tensor)
    activations = [out[layer_to_use][:, feature_to_use:feature_to_use+1, :] for layer_to_use, feature_to_use in zip(layers_to_use, features_to_use)]

    losses = []
    for layer_activation in activations:
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()

    grad = input_tensor.grad.data
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)

    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / (g_std + 1e-5)

    input_tensor.data += config['lr'] * smooth_grad
    input_tensor.grad.data.zero_()

"""
Setup Model Validation
"""
model = fetch_and_prepare_model(config['model_name'], config['pretrained_weights'])

model_test_input = torch.zeros((1, 3, 128, 128)).to(DEVICE)
if DEVICE.type == 'cuda':
    model_test_input = model_test_input.half()
    
model_test_output, model_layer_outputs = model(model_test_input)

layer_names = model.layer_names
feature_counts = []
for layer_name, layer_output in model_layer_outputs.items():
    feature_counts.append(layer_output.shape[1])

"""
OSC Receiver Setup
"""
def osc_set_layer(address, *args):
    layer = str(args[0])
    if layer in layer_names:
        config["layers_to_use"] = [layer]

def osc_set_feature(address, *args):
    feature = int(args[0])
    layer = config["layers_to_use"][0]
    layer_index = layer_names.index(layer)
    if 0 <= feature < feature_counts[layer_index]:
        config["features_to_use"] = [feature]

def osc_set_pyramid_size(address, *args):
    config["pyramid_size"] = int(args[0])

def osc_set_pyramid_ratio(address, *args):
    config["pyramid_ratio"] = float(args[0])

def osc_set_gradient_iterations(address, *args):
    config["num_gradient_ascent_iterations"] = int(args[0])

def osc_set_learning_rate(address, *args):
    config["lr"] = float(args[0])

def osc_set_image_blend(address, *args):
    config["image_blend_factor"] = float(args[0])

def osc_set_bg_subtraction(address, *args):
    config["use_bg_subtraction"] = bool(args[0])

osc_dispatcher = dispatcher.Dispatcher()
osc_dispatcher.map("/deepdream/layer", osc_set_layer)
osc_dispatcher.map("/deepdream/feature", osc_set_feature)
osc_dispatcher.map("/deepdream/pyramid_size", osc_set_pyramid_size)
osc_dispatcher.map("/deepdream/pyramid_ratio", osc_set_pyramid_ratio)
osc_dispatcher.map("/deepdream/iterations", osc_set_gradient_iterations)
osc_dispatcher.map("/deepdream/learning_rate", osc_set_learning_rate)
osc_dispatcher.map("/deepdream/blend", osc_set_image_blend)
osc_dispatcher.map("/deepdream/bg_subtraction", osc_set_bg_subtraction)

osc_server_instance = osc_server.ThreadingOSCUDPServer((osc_receive_ip, osc_receive_port), osc_dispatcher)
osc_thread = None

def start_osc_server():
    osc_server_instance.serve_forever()

def osc_start():
    global osc_thread
    if osc_thread is None or not osc_thread.is_alive():
        osc_thread = threading.Thread(target=start_osc_server, daemon=True)
        osc_thread.start()
        
def osc_stop():
    global osc_thread
    osc_server_instance.shutdown()
    osc_server_instance.server_close()
    if osc_thread is not None and osc_thread.is_alive():
        osc_thread.join()

"""
Real Time version with camera input
"""
def setup_video_capture(camera_index, camera_resolution):
    camera = cv.VideoCapture(camera_index)
    camera.set(3, camera_resolution[0])
    camera.set(4, camera_resolution[1])
    return camera

def finish_video_capture(camera):
    camera.release()

def capture_image(camera, target_resolution):
    ret, camera_frame = camera.read()
    if not ret:
        return None
    
    if target_resolution is not None:
        if isinstance(target_resolution, int) and target_resolution != -1:
            current_height, current_width = camera_frame.shape[:2]
            new_width = target_resolution
            new_height = int(current_height * (new_width / current_width))
            camera_frame = cv.resize(camera_frame, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            camera_frame = cv.resize(camera_frame, (target_resolution[1], target_resolution[0]), interpolation=cv.INTER_CUBIC)

    camera_frame = camera_frame.astype(np.float32)
    camera_frame /= 255.0
    return camera_frame

def apply_deep_dream(proc_image, camera_image):
    if proc_image is not None:
        proc_image = proc_image * (1.0 - config["image_blend_factor"]) + camera_image * config["image_blend_factor"]
        proc_image = deep_dream_static_image(config, proc_image)
    else:
        proc_image = deep_dream_static_image(config, camera_image)
    return proc_image

"""
PyQt5 GUI Integration
"""
class DeepDreamGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # Background Subtraction variables
        self.reference_background = None
        self.bg_threshold = 0.15
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('DeepDream Controller')
        main_layout = QHBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, stretch=1)
        
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 0, 10, 0)
        
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel('Layer:'))
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(layer_names)
        self.layer_combo.setCurrentText(config["layers_to_use"][0])
        self.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        controls_layout.addLayout(layer_layout)
        
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(QLabel('Feature Map:'))
        self.feature_combo = QComboBox()
        self.update_feature_combo(config["layers_to_use"][0])
        self.feature_combo.currentIndexChanged.connect(self.on_feature_changed)
        feature_layout.addWidget(self.feature_combo)
        controls_layout.addLayout(feature_layout)
        
        pyr_size_layout = QHBoxLayout()
        pyr_size_layout.addWidget(QLabel('Pyramid Size:'))
        self.pyr_size_spin = QSpinBox()
        self.pyr_size_spin.setRange(1, 10)
        self.pyr_size_spin.setValue(int(config["pyramid_size"]))
        self.pyr_size_spin.valueChanged.connect(self.on_pyr_size_changed)
        pyr_size_layout.addWidget(self.pyr_size_spin)
        controls_layout.addLayout(pyr_size_layout)
        
        pyr_ratio_layout = QHBoxLayout()
        pyr_ratio_layout.addWidget(QLabel('Pyramid Ratio:'))
        self.pyr_ratio_spin = QDoubleSpinBox()
        self.pyr_ratio_spin.setRange(1.0, 5.0)
        self.pyr_ratio_spin.setSingleStep(0.1)
        self.pyr_ratio_spin.setValue(float(config["pyramid_ratio"]))
        self.pyr_ratio_spin.valueChanged.connect(self.on_pyr_ratio_changed)
        pyr_ratio_layout.addWidget(self.pyr_ratio_spin)
        controls_layout.addLayout(pyr_ratio_layout)
        
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel('Iterations:'))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(int(config["num_gradient_ascent_iterations"]))
        self.iter_spin.valueChanged.connect(self.on_iter_changed)
        iter_layout.addWidget(self.iter_spin)
        controls_layout.addLayout(iter_layout)
        
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel('Learning Rate:'))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setDecimals(3)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(float(config["lr"]))
        self.lr_spin.valueChanged.connect(self.on_lr_changed)
        lr_layout.addWidget(self.lr_spin)
        controls_layout.addLayout(lr_layout)
        
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel('Blending Factor:'))
        self.blend_spin = QDoubleSpinBox()
        self.blend_spin.setRange(0.0, 1.0)
        self.blend_spin.setSingleStep(0.05)
        self.blend_spin.setValue(float(config["image_blend_factor"]))
        self.blend_spin.valueChanged.connect(self.on_blend_changed)
        blend_layout.addWidget(self.blend_spin)
        controls_layout.addLayout(blend_layout)

        bg_layout = QHBoxLayout()
        self.bg_sub_cb = QCheckBox("Mask Background")
        self.bg_sub_cb.setChecked(config["use_bg_subtraction"])
        self.bg_sub_cb.stateChanged.connect(self.on_bg_sub_changed)
        bg_layout.addWidget(self.bg_sub_cb)
        
        self.bg_reset_btn = QPushButton("Reset BG")
        self.bg_reset_btn.clicked.connect(self.reset_background)
        bg_layout.addWidget(self.bg_reset_btn)
        controls_layout.addLayout(bg_layout)
        
        controls_layout.addStretch()
        
        self.exit_btn = QPushButton('Exit')
        self.exit_btn.clicked.connect(self.close_app)
        controls_layout.addWidget(self.exit_btn)
        
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        if use_live_camera_input:
            self.camera = setup_video_capture(0, image_resolution)
        else:
            self.camera = setup_video_capture(movie_file_path, image_resolution)
            
        osc_start()
        self.proc_image = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(1)

    def update_feature_combo(self, layer):
        self.feature_combo.blockSignals(True)
        self.feature_combo.clear()

        if layer in layer_names:
            layer_index = layer_names.index(layer)
            feature_count = feature_counts[layer_index]

            self.feature_combo.addItems([str(i) for i in range(feature_count)])

            current_feature = int(config["features_to_use"][0])
            current_feature = max(0, min(current_feature, feature_count - 1))
            config["features_to_use"][0] = current_feature

            self.feature_combo.setCurrentIndex(current_feature)

        self.feature_combo.blockSignals(False)

    def on_layer_changed(self, layer):
        config["layers_to_use"] = [layer]
        self.update_feature_combo(layer)
        
    def on_feature_changed(self, idx):
        if idx >= 0:
            config["features_to_use"] = [idx]
            
    def on_pyr_size_changed(self, val):
        config["pyramid_size"] = val
        
    def on_pyr_ratio_changed(self, val):
        config["pyramid_ratio"] = val
        
    def on_iter_changed(self, val):
        config["num_gradient_ascent_iterations"] = val
        
    def on_lr_changed(self, val):
        config["lr"] = val
        
    def on_blend_changed(self, val):
        config["image_blend_factor"] = val

    def on_bg_sub_changed(self, state):
        config["use_bg_subtraction"] = (state == Qt.Checked)
        if config["use_bg_subtraction"]:
            self.reset_background()

    def reset_background(self):
        self.reference_background = None
        print("Background snapshot requested...")

    def sync_gui_with_osc(self):
        self.layer_combo.blockSignals(True)
        self.feature_combo.blockSignals(True)
        self.pyr_size_spin.blockSignals(True)
        self.pyr_ratio_spin.blockSignals(True)
        self.iter_spin.blockSignals(True)
        self.lr_spin.blockSignals(True)
        self.blend_spin.blockSignals(True)
        self.bg_sub_cb.blockSignals(True)

        try:
            osc_layer = config["layers_to_use"][0]
            osc_feature = config["features_to_use"][0]

            if self.layer_combo.currentText() != osc_layer:
                self.layer_combo.setCurrentText(osc_layer)
                self.update_feature_combo(osc_layer)

            if osc_layer in layer_names:
                layer_index = layer_names.index(osc_layer)
                max_feature = feature_counts[layer_index] - 1
                osc_feature = max(0, min(int(osc_feature), max_feature))

                if config["features_to_use"][0] != osc_feature:
                    config["features_to_use"][0] = osc_feature

                if self.feature_combo.count() != feature_counts[layer_index]:
                    self.update_feature_combo(osc_layer)

                if self.feature_combo.currentIndex() != osc_feature:
                    self.feature_combo.setCurrentIndex(osc_feature)

            if self.pyr_size_spin.value() != int(config["pyramid_size"]):
                self.pyr_size_spin.setValue(int(config["pyramid_size"]))

            if self.pyr_ratio_spin.value() != float(config["pyramid_ratio"]):
                self.pyr_ratio_spin.setValue(float(config["pyramid_ratio"]))

            if self.iter_spin.value() != int(config["num_gradient_ascent_iterations"]):
                self.iter_spin.setValue(int(config["num_gradient_ascent_iterations"]))

            if self.lr_spin.value() != float(config["lr"]):
                self.lr_spin.setValue(float(config["lr"]))

            if self.blend_spin.value() != float(config["image_blend_factor"]):
                self.blend_spin.setValue(float(config["image_blend_factor"]))
                
            if self.bg_sub_cb.isChecked() != config["use_bg_subtraction"]:
                self.bg_sub_cb.setChecked(config["use_bg_subtraction"])

        finally:
            self.layer_combo.blockSignals(False)
            self.feature_combo.blockSignals(False)
            self.pyr_size_spin.blockSignals(False)
            self.pyr_ratio_spin.blockSignals(False)
            self.iter_spin.blockSignals(False)
            self.lr_spin.blockSignals(False)
            self.blend_spin.blockSignals(False)
            self.bg_sub_cb.blockSignals(False)

    def process_frame(self):
        self.sync_gui_with_osc()
        camera_frame = capture_image(self.camera, None)
        
        if camera_frame is None:
            self.close_app()
            return
            
        # --- Apply Static Background Subtraction ---
        if config.get("use_bg_subtraction", False):
            # 1. Take a snapshot if we don't have one yet
            if self.reference_background is None:
                self.reference_background = camera_frame.copy()
                print("Background snapshot captured.")
            
            # 2. Calculate absolute difference between live frame and the reference snapshot
            diff = cv.absdiff(self.reference_background, camera_frame)
            
            # Convert difference to grayscale (single channel)
            gray_diff = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
            
            # 3. Apply a threshold
            _, fg_mask_norm = cv.threshold(gray_diff, self.bg_threshold, 1.0, cv.THRESH_BINARY)
            
            # Clean up the mask slightly to remove sensor noise
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            fg_mask_norm = cv.morphologyEx(fg_mask_norm, cv.MORPH_OPEN, kernel)
            fg_mask_norm = cv.morphologyEx(fg_mask_norm, cv.MORPH_CLOSE, kernel)
            
            # Smooth the edges of the mask
            fg_mask_norm = cv.GaussianBlur(fg_mask_norm, (11, 11), 0)
            
            # 4. Multiply it to mask out the background in the camera feed
            camera_frame = camera_frame * np.expand_dims(fg_mask_norm, axis=-1)
        
        # --- Continue with DeepDream ---
        self.proc_image = apply_deep_dream(self.proc_image, camera_frame)
        
        proc_image_cv = self.proc_image * 255.0
        proc_image_cv = np.ascontiguousarray(proc_image_cv.astype(np.uint8))

        h, w, ch = proc_image_cv.shape
        bytes_per_line = ch * w
        q_img = QImage(proc_image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
            
    def close_app(self):
        self.timer.stop()
        osc_stop()
        finish_video_capture(self.camera)
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DeepDreamGUI()
    ex.show()
    sys.exit(app.exec_())