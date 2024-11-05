import monai.transforms as mt
from monai.utils.enums import InterpolateMode

from openood.utils.config import Config


class Med3DPreprocessor(mt.Compose):
    def __init__(self, config: Config):
        assert config.dataset.interpolation in list(InterpolateMode)
        self.interpolation = config.dataset.interpolation
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        transforms = [
            mt.EnsureChannelFirst(),
            # Reorient to RAS (Right-Anterior-Superior) orientation
            mt.Orientation(axcodes='RAS'),
            # Apply z-score normalization (mean 0, std 1)
            mt.NormalizeIntensity(nonzero=True, channel_wise=True),
            mt.Resize(spatial_size=tuple(self.pre_size),
                      mode=self.interpolation),
            # Small random affine (rotations & scaling) transformations
            mt.RandAffine(
                prob=0.5,
                rotate_range=(0.087, 0.087, 0.087),  # Small rotations (~5 deg)
                scale_range=(0.05, 0.05, 0.05),  # Small scaling (0.95 - 1.05)
                padding_mode='border'),
            # Add subtle elastic deformation to simulate anatomical variability
            mt.Rand3DElastic(prob=0.3,
                             sigma_range=(8, 10),
                             magnitude_range=(1, 5)),
            mt.RandSpatialCrop(roi_size=tuple(self.image_size),
                               random_size=False),
            mt.RandShiftIntensity(offsets=0.1, prob=0.5),
            mt.RandGaussianNoise(mean=0.0, std=0.01, prob=0.5)
        ]
        super().__init__(transforms)

    def setup(self, **kwargs):
        pass


class Med3DTestPreprocessor(mt.Compose):
    def __init__(self, config: Config):
        assert config.dataset.interpolation in list(InterpolateMode)
        self.interpolation = config.dataset.interpolation
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        transforms = [
            mt.EnsureChannelFirst(),
            # Reorient to RAS (Right-Anterior-Superior) orientation
            mt.Orientation(axcodes='RAS'),
            # Apply z-score normalization (mean 0, std 1)
            mt.NormalizeIntensity(nonzero=True, channel_wise=True),
            mt.Resize(spatial_size=tuple(self.image_size),
                      mode=self.interpolation),
        ]
        super().__init__(transforms)

    def setup(self, **kwargs):
        pass
