from .resnet import *
from .vgg import *

import torch
from torch import nn
from torchvision import models


def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = vgg16_bn(num_classes=num_classes)  # NOTE: must use the BN version, otherwise accuracy is poor
    else:
        raise ValueError(f"Unknown model: {model_name} (only 'resnet18' and 'vgg16' are supported)")
    return model


def load_model(model_path, model_name, num_classes, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ckpt = torch.load(model_path, map_location=device)
    if isinstance(model_ckpt, dict):
        if 'state_dict' in model_ckpt and isinstance(model_ckpt['state_dict'], dict):
            inner_classes = model_ckpt.get('config', {}).get('num_classes') if isinstance(model_ckpt.get('config'), dict) else None
            if inner_classes is not None:
                num_classes = inner_classes
            model_ckpt = model_ckpt['state_dict']
        model = get_model(model_name, num_classes)
        renamed = {}
        for k, v in model_ckpt.items():
            k = k.replace('module.', '')
            if k.startswith('fc_layer.'):
                k = 'fc.' + k[len('fc_layer.'):]
            renamed[k] = v
        model.load_state_dict(renamed)
    else:
        model = model_ckpt
    model = model.to(device)
    return model
