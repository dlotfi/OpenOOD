import os

import torch
from numpy import load
from torch.utils.data import DataLoader
from monai.data import DataLoader as MonaiDataLoader

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .imglist_augmix_dataset import ImglistAugMixDataset
from .imglist_extradata_dataset import (ImglistExtraDataDataset,
                                        TwoSourceSampler)
from .med3d_imglist_dataset import Med3DImglistDataset
from .udg_dataset import UDGDataset

__all__ = (
    'ImglistDataset',
    'UDGDataset',
)


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config) \
            if split_config.dataset_class != 'Med3DImglistDataset' else None

        if split_config.dataset_class == 'ImglistExtraDataDataset':
            dataset = ImglistExtraDataDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor,
                extra_data_pth=split_config.extra_data_pth,
                extra_label_pth=split_config.extra_label_pth,
                extra_percent=split_config.extra_percent)

            batch_sampler = TwoSourceSampler(dataset.orig_ids,
                                             dataset.extra_ids,
                                             split_config.batch_size,
                                             split_config.orig_ratio)

            dataloader = DataLoader(dataset,
                                    batch_sampler=batch_sampler,
                                    num_workers=dataset_config.num_workers,
                                    drop_last=bool(split_config.drop_last))
        elif split_config.dataset_class == 'ImglistAugMixDataset':
            dataset = ImglistAugMixDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler,
                                    drop_last=bool(split_config.drop_last))
        elif split_config.dataset_class == 'Med3DImglistDataset':
            dataset = Med3DImglistDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                num_channels=dataset_config.num_channels,
                preprocessor=preprocessor)
            dataloader = MonaiDataLoader(
                dataset=dataset,
                batch_size=split_config.batch_size,
                shuffle=split_config.shuffle,
                num_workers=dataset_config.num_workers,
                drop_last=bool(split_config.drop_last),
                pin_memory=torch.cuda.is_available())
        else:
            CustomDataset = eval(split_config.dataset_class)
            dataset = CustomDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                num_channels=dataset_config.num_channels,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler,
                                    drop_last=bool(split_config.drop_last))

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)

    def _get_loader(name, imglist_pth, data_dir, preprocessor,
                    data_aux_preprocessor):
        if ood_config.dataset_class == 'Med3DImglistDataset':
            dataset = Med3DImglistDataset(name=name,
                                          imglist_pth=imglist_pth,
                                          data_dir=data_dir,
                                          num_classes=ood_config.num_classes,
                                          num_channels=ood_config.num_channels,
                                          preprocessor=preprocessor)
            dataloader = MonaiDataLoader(dataset=dataset,
                                         batch_size=ood_config.batch_size,
                                         shuffle=ood_config.shuffle,
                                         num_workers=ood_config.num_workers,
                                         drop_last=bool(ood_config.drop_last),
                                         pin_memory=torch.cuda.is_available())
        else:
            dataset = CustomDataset(
                name=name,
                imglist_pth=imglist_pth,
                data_dir=data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
        return dataloader

    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config) \
            if ood_config.dataset_class != 'Med3DImglistDataset' else None
        if split == 'val':
            # validation set
            dataloader_dict[split] = _get_loader(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                sub_dataloader_dict[dataset_name] = _get_loader(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader


def get_feature_opengan_dataloader(dataset_config: Config):
    feat_root = dataset_config.feat_root

    dataloader_dict = {}
    for d in ['id_train', 'id_val', 'ood_val']:
        # load in the cached feature
        loaded_data = load(os.path.join(feat_root, f'{d}.npz'),
                           allow_pickle=True)
        total_feat = torch.from_numpy(loaded_data['feat_list'])
        total_labels = loaded_data['label_list']
        del loaded_data
        # reshape the vector to fit in to the network
        total_feat.unsqueeze_(-1).unsqueeze_(-1)
        # let's see what we got here should be something like:
        # torch.Size([total_num, channel_size, 1, 1])
        print('Loaded feature size: {}'.format(total_feat.shape))

        if d == 'id_train':
            split_config = dataset_config['train']
        else:
            split_config = dataset_config['val']

        dataset = FeatDataset(feat=total_feat, labels=total_labels)
        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)
        dataloader_dict[d] = dataloader

    return dataloader_dict


def _get_feat_loader(feat_root: str, filename: str, batch_size: int,
                     z_normalize_features: bool, shuffle: bool,
                     drop_last: bool, num_workers: int) -> DataLoader:
    # load in the cached feature
    loaded_data = load(os.path.join(feat_root, f'{filename}.npz'),
                       allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    total_labels = loaded_data['label_list']
    if torch.isnan(total_feat).any() or torch.isinf(total_feat).any():
        print('NaN or Inf detected in the feature')
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))
    # z-score normalize the feature
    if z_normalize_features:
        mean = total_feat.mean(dim=0)
        std = total_feat.std(dim=0)
        total_feat = (total_feat - mean) / (std + 1e-6)
        print('Features have bean z-score normalized ' 'in each dimension')

    dataset = FeatDataset(feat=total_feat, labels=total_labels)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader


def get_feature_nflow_dataloader(dataset_config: Config):
    feat_root = dataset_config.feat_root
    z_normalize_features = bool(dataset_config.z_normalize_feat)

    dataloader_dict = {}
    for d in ['id_train', 'id_val', 'ood_val']:
        if d == 'id_train':
            split_config = dataset_config['train']
        else:
            split_config = dataset_config['val']
        dataloader_dict[d] = _get_feat_loader(
            feat_root,
            filename=d,
            batch_size=split_config.batch_size,
            z_normalize_features=z_normalize_features,
            shuffle=bool(split_config.shuffle),
            drop_last=bool(split_config.drop_last),
            num_workers=dataset_config.num_workers)

    return dataloader_dict


def get_feature_nflow_test_dataloaders(dataset_config: Config,
                                       ood_dataset_config: Config,
                                       load_train_val: bool = True):
    ood_feat_root = ood_dataset_config.feat_root
    id_dataloader_dict = {}
    ood_dataloader_dict = {}

    id_z_normalize_features = bool(dataset_config.z_normalize_feat)
    ood_z_normalize_features = bool(ood_dataset_config.z_normalize_feat)
    if id_z_normalize_features ^ ood_z_normalize_features:
        raise ValueError('z_normalize_feat should be set to True for '
                         'all or none of the datasets.')

    if load_train_val:
        id_train_dataloader_dict = get_feature_nflow_dataloader(dataset_config)
        id_dataloader_dict['train'] = id_train_dataloader_dict['id_train']
        id_dataloader_dict['val'] = id_train_dataloader_dict['id_val']
        if 'val' in ood_dataset_config.split_names:
            ood_dataloader_dict['val'] = id_train_dataloader_dict['ood_val']

    test_config = dataset_config['test']
    id_dataloader_dict['test'] = _get_feat_loader(
        ood_feat_root,
        filename=dataset_config.name,
        batch_size=test_config.batch_size,
        z_normalize_features=id_z_normalize_features,
        shuffle=bool(test_config.shuffle),
        drop_last=bool(test_config.drop_last),
        num_workers=dataset_config.num_workers)

    ood_batch_size = ood_dataset_config.batch_size
    ood_shuffle = bool(ood_dataset_config.shuffle)
    ood_drop_last = bool(ood_dataset_config.drop_last)
    ood_num_workers = ood_dataset_config.num_workers

    for split in set(ood_dataset_config.split_names) - {'val'}:
        ood_dataloader_dict[split] = {}
        for d in ood_dataset_config[split].datasets:
            ood_dataloader_dict[split][d] = _get_feat_loader(
                ood_feat_root,
                filename=d,
                batch_size=ood_batch_size,
                z_normalize_features=ood_z_normalize_features,
                shuffle=ood_shuffle,
                drop_last=ood_drop_last,
                num_workers=ood_num_workers)

    return id_dataloader_dict, ood_dataloader_dict
