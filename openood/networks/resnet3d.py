import re
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import ResNet, ResNetBlock, ResNetBottleneck
from monai.networks.nets.resnet import (get_inplanes,
                                        get_medicalnet_pretrained_resnet_args,
                                        get_pretrained_resnet_medicalnet)


class ResNet3D(ResNet):
    def __init__(self, block, layers, block_inplanes, **kwargs):
        super(ResNet3D, self).__init__(block,
                                       layers,
                                       block_inplanes,
                                       spatial_dims=3,
                                       **kwargs)
        self.feature_size = block_inplanes[3] * block.expansion

    def forward(self,
                x: torch.Tensor,
                return_feature: bool = False,
                return_feature_list: bool = False) -> torch.Tensor:
        feature1 = self.act(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x: torch.Tensor,
                          threshold: float) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = x.view(x.size(0), -1)
        logits_cls = self.fc(x)
        return logits_cls

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc


def _resnet(
    arch: str,
    block: type[ResNetBlock | ResNetBottleneck],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool | str,
    progress: bool,
    **kwargs: Any,
) -> ResNet3D:
    model = ResNet3D(block, layers, block_inplanes, **kwargs)
    if pretrained:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(pretrained, str):
            if Path(pretrained).exists():
                print(f'Loading weights from {pretrained}...')
                model_state_dict = torch.load(pretrained, map_location=device)
            else:
                # Throw error
                raise FileNotFoundError(
                    'The pretrained checkpoint file is not found')
        else:
            # Also check bias downsample and shortcut.
            if kwargs.get('spatial_dims', 3) == 3 and \
               kwargs.get('n_input_channels', 3) == 1:
                search_res = re.search(r'resnet(\d+)', arch)
                if search_res:
                    resnet_depth = int(search_res.group(1))
                else:
                    raise ValueError(
                        "arch argument should be as 'resnet_{resnet_depth}")

                # Check model bias_downsample and shortcut_type
                bias_downsample, shortcut_type = (
                    get_medicalnet_pretrained_resnet_args(resnet_depth))
                if shortcut_type == kwargs.get(
                        'shortcut_type',
                        'B') and (bias_downsample == kwargs.get(
                            'bias_downsample', True)):
                    # Download the MedicalNet pretrained model
                    model_state_dict = get_pretrained_resnet_medicalnet(
                        resnet_depth, device=device, datasets23=True)
                else:
                    raise NotImplementedError(
                        f'Please set shortcut_type to {shortcut_type} and '
                        f'bias_downsample to {bias_downsample} when using '
                        f'pretrained MedicalNet resnet{resnet_depth}')
            else:
                raise NotImplementedError('MedicalNet pretrained weights are '
                                          'only available for 3D models with a'
                                          ' single channel input')
        model_state_dict = {
            key.replace('module.', ''): value
            for key, value in model_state_dict.items()
        }
        model.load_state_dict(model_state_dict, strict=False)
    return model


def ResNet3D_18(num_classes: int,
                in_channels: int = 1,
                pretrained: bool = False,
                **kwargs: Any) -> ResNet3D:
    """ResNet-18 with optional pretrained support.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis
    <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        num_classes (int): Number of classes for the classification head.
        in_channels (int): Number of input channels.
        pretrained (bool): If True, returns a model pretrained on 23
                           medical datasets
    """
    return _resnet('resnet18',
                   ResNetBlock, [2, 2, 2, 2],
                   get_inplanes(),
                   pretrained,
                   progress=True,
                   n_input_channels=in_channels,
                   num_classes=num_classes,
                   shortcut_type='A',
                   bias_downsample=True,
                   **kwargs)


def ResNet3D_50(num_classes: int,
                in_channels: int = 1,
                pretrained: bool = False,
                **kwargs: Any) -> ResNet3D:
    """ResNet-50 with optional pretrained support.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis
    <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        num_classes (int): Number of classes for the classification head.
        in_channels (int): Number of input channels.
        pretrained (bool): If True, returns a model pretrained on 23
                           medical datasets
    """
    return _resnet('resnet50',
                   ResNetBottleneck, [3, 4, 6, 3],
                   get_inplanes(),
                   pretrained,
                   progress=True,
                   n_input_channels=in_channels,
                   num_classes=num_classes,
                   shortcut_type='B',
                   bias_downsample=False,
                   **kwargs)
