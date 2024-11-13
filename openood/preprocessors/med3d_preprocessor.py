import monai.transforms as mt

from openood.utils.config import Config


class Med3DPreprocessor(mt.Compose):
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        # All brain images in this collection of datasets are registered to
        # the SRI24 atlas, resampled to an isotropic resolution of 1mm³,
        # brain-extracted, and normalized (percentile clip to 0.1 - 99.9)
        # For non-brain images, the preprocessing includes resampling to
        # 1mm³, center cropping to 240x240x155, and normalization (same as
        # brain images)
        if config.dataset.processing_type == 'medood':
            transforms = [
                mt.EnsureChannelFirst(),
                # Reorient to RAS (Right-Anterior-Superior) orientation
                mt.Orientation(axcodes='RAS'),
                # Apply z-score normalization (mean 0, std 1)
                mt.NormalizeIntensity(nonzero=True, channel_wise=True),
                mt.Resize(spatial_size=tuple(self.pre_size)),
                mt.RandSpatialCrop(roi_size=tuple(self.image_size),
                                   random_size=False),
            ]
        elif config.dataset.processing_type == 'medood-strong':
            transforms = [
                mt.EnsureChannelFirst(),
                # Reorient to RAS (Right-Anterior-Superior) orientation
                mt.Orientation(axcodes='RAS'),
                # Apply z-score normalization (mean 0, std 1)
                mt.NormalizeIntensity(nonzero=True, channel_wise=True),
                mt.Resize(spatial_size=tuple(self.pre_size)),
                # Small random affine (rotations & scaling) transformations
                mt.RandAffine(
                    prob=0.5,
                    rotate_range=(0.087, 0.087, 0.087),  # ~5 deg
                    scale_range=(0.05, 0.05, 0.05),
                    padding_mode='border'),
                # Subtle elastic deformation to simulate anatomical variability
                mt.Rand3DElastic(prob=0.3,
                                 sigma_range=(8, 10),
                                 magnitude_range=(1, 5)),
                mt.RandSpatialCrop(roi_size=tuple(self.image_size),
                                   random_size=False),
                mt.RandShiftIntensity(prob=0.5, offsets=0.1),
                mt.RandGaussianNoise(prob=0.5, mean=0.0, std=0.01),
                mt.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
            ]
        else:
            raise ValueError(f'Unknown processing type: '
                             f'{config.dataset.processing_type}')
        super().__init__(transforms)

    def setup(self, **kwargs):
        pass


class Med3DTestPreprocessor(mt.Compose):
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        if config.dataset.processing_type.startswith('medood'):
            transforms = [
                mt.EnsureChannelFirst(),
                # Reorient to RAS (Right-Anterior-Superior) orientation
                mt.Orientation(axcodes='RAS'),
                # Apply z-score normalization (mean 0, std 1)
                mt.NormalizeIntensity(nonzero=True, channel_wise=True),
                mt.Resize(spatial_size=tuple(self.image_size)),
            ]
        else:
            raise ValueError(f'Unknown processing type: '
                             f'{config.dataset.processing_type}')
        super().__init__(transforms)

    def setup(self, **kwargs):
        pass
