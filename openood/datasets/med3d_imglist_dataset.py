import ast
import logging
import os
from typing import Callable, Any

import numpy as np
import torch
from monai.config import DtypeLike
from monai.data import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import get_seed, MAX_SEED

from .base_dataset import BaseDataset


class Med3DImglistDataset(BaseDataset, Randomizable):
    def __init__(self,
                 name: str,
                 imglist_pth: str,
                 data_dir: str,
                 num_classes: int,
                 preprocessor: Callable,
                 maxlen: int = None,
                 dummy_read: bool = False,
                 dummy_size: bool = None,
                 num_channels: int = 1,
                 image_only: bool = True,
                 transform_with_metadata: bool = False,
                 dtype: DtypeLike = np.float32,
                 reader: ImageReader | str | None = None,
                 *args,
                 **kwargs):
        super(Med3DImglistDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')
        if image_only and transform_with_metadata:
            raise ValueError(
                'transform_with_metadata=True requires image_only=False.')
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def randomize(self, data: Any | None = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype='uint32')

    def getitem(self, index):
        self.randomize()
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except Exception as e:
            logging.warning(
                f'Encountered an error during preprocessor setup: {e}')

        try:
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                # load data and optionally meta
                if self.image_only:
                    image = self.loader(path)
                else:
                    image, meta_data = self.loader(path)

                # apply the transforms
                if isinstance(self.transform_image, Randomizable):
                    self.transform_image.set_random_state(seed=self._seed)

                if self.transform_with_metadata:
                    image, meta_data = apply_transform(self.transform_image,
                                                       (image, meta_data),
                                                       map_items=False,
                                                       unpack_items=True)
                    sample['meta_data'] = meta_data
                else:
                    image = apply_transform(self.transform_image,
                                            image,
                                            map_items=False)

                sample['data'] = image
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
