import os
import random
from functools import wraps
from typing import Callable, Tuple

import SimpleITK as sitk
import ants
import numpy as np
import torchio as tio

TransformFunc = Callable[[sitk.Image], Tuple[sitk.Image, dict]]


def sitk_to_torchio(func):
    @wraps(func)
    def wrapper(image: sitk.Image, *args, **kwargs) -> Tuple[sitk.Image, dict]:
        torchio_image = tio.Image.from_sitk(image)
        torchio_subject = tio.Subject(image=torchio_image)
        result = func(torchio_subject, *args, **kwargs)
        if isinstance(result, Tuple):
            torchio_image, transform_details = result
            return torchio_image.as_sitk(), transform_details
        torchio_image = result['image']
        transform_details = dict()
        for i, transform in enumerate(result.history, start=1):
            transform_name = f'{i}_{transform.__class__.__name__}'
            transform_details[transform_name] = dict()
            for arg_name in transform.args_names:
                arg_value = getattr(transform, arg_name)
                if isinstance(arg_value, dict):
                    arg_value = arg_value['image']
                transform_details[transform_name][arg_name] = str(arg_value)
        return torchio_image.as_sitk(), transform_details

    return wrapper


@sitk_to_torchio
def motion_artifact(subject: tio.Subject) -> tio.Subject:
    motion_transform = tio.RandomMotion(
        degrees=10,  # Rotation range [-10, 10] degrees
        translation=10  # Translation range [-10, 10] mm
    )
    return motion_transform(subject)


@sitk_to_torchio
def ghost_artifact(subject: tio.Subject) -> tio.Subject:
    # Randomly select one or more axes
    axes = random.sample([0, 1, 2], k=random.randint(1, 3))
    multi_transforms = []
    for axis in axes:
        ghost_transform = tio.RandomGhosting(
            num_ghosts=(1, 3),  # Number of ghosted copies
            axes=axis,  # Axis along which the ghosting effect is applied
            intensity=(0.4, 0.6)  # Intensity range of the ghosting effect
        )

        multi_transforms.append(ghost_transform)
    multi_transforms = tio.Compose(multi_transforms)
    return multi_transforms(subject)


@sitk_to_torchio
def bias_artifact(subject: tio.Subject) -> tio.Subject:
    bias_transform = tio.RandomBiasField(
        coefficients=1  # Coefficient for the polynomial basis functions
    )
    return bias_transform(subject)


@sitk_to_torchio
def spike_artifact(subject: tio.Subject) -> tio.Subject:
    spike_transform = tio.RandomSpike(
        num_spikes=1,  # Number of spikes to add
        intensity=(0.2, 0.5)  # Intensity range of the spikes
    )
    return spike_transform(subject)


@sitk_to_torchio
def gaussian_noise(subject: tio.Subject) -> tio.Subject:
    # Apply z-normalization
    z_normalize = tio.ZNormalization()
    # Add Gaussian noise
    noise_transform = tio.RandomNoise(
        mean=0,  # Mean of the Gaussian noise
        std=0.5  # Standard deviation of the Gaussian noise
    )
    multi_transforms = tio.Compose([z_normalize, noise_transform])
    return multi_transforms(subject)


@sitk_to_torchio
def downsampling(subject: tio.Subject) -> tio.Subject:
    original_spacing = subject['image'].spacing
    # Define the downsampling factors for xy and z axes, ensuring that
    # at least one axis is downsampled
    while True:
        downsample_factor_xy = random.choice([1, 2, 3])
        downsample_factor_z = random.choice([1, 2, 3])
        if not (downsample_factor_xy == 1 and downsample_factor_z == 1):
            break
    # Downsample the image
    downsample_transform = tio.Resample(
        target=(original_spacing[0] * downsample_factor_xy,
                original_spacing[1] * downsample_factor_xy,
                original_spacing[2] * downsample_factor_z))
    # Upsample back to the original resolution
    upsample_transform = tio.Resample(target=original_spacing)
    multi_transforms = tio.Compose([downsample_transform, upsample_transform])
    return multi_transforms(subject)


@sitk_to_torchio
def scaling_perturbation(subject: tio.Subject) -> tio.Subject:
    # Double the size in half of the cases, otherwise shrink by half
    scales = 2 if random.random() < 0.5 else 0.5
    scaling_transform = tio.Affine(
        scales=scales,  # Scaling factor
        degrees=0,  # No rotation
        translation=0,  # No translation
    )
    return scaling_transform(subject)


@sitk_to_torchio
def gamma_alteration(subject: tio.Subject) -> tio.Subject:
    log_gamma = random.choice([-1.5, 1.5])
    gamma_transform = tio.RandomGamma(log_gamma=(log_gamma, log_gamma))
    return gamma_transform(subject)


@sitk_to_torchio
def truncation(subject: tio.Subject) -> Tuple[tio.Image, dict]:
    image = subject['image']
    # Get the original size of the image
    original_shape = np.array(image.shape)

    # Randomly choose a direction (axis) to truncate
    # 0-axis is the channel dimension
    axis = random.choice([1, 2, 3])

    # Calculate the truncation size
    truncation_shape = original_shape.copy()
    truncation_shape[axis] = truncation_shape[axis] // 2

    # Randomly choose to truncate from the start or end
    start_index = 0 if random.random() < 0.5 else \
        (original_shape[axis] - truncation_shape[axis])

    # Create slices for truncation
    slices = [slice(None)] * 4
    slices[axis] = slice(start_index, start_index + truncation_shape[axis])

    # Create a copy of the image data
    truncated_image_data = image.data.clone()

    # Fill the truncated area with zeros
    truncated_image_data[slices] = 0

    # Create a new TorchIO image with the modified data
    truncated_image = tio.Image(tensor=truncated_image_data,
                                affine=image.affine)

    transform_details = {
        '1_Truncation': {
            'axis': str(axis),
            'start_index': str(start_index),
            'truncation_shape': str(truncation_shape)
        }
    }
    return truncated_image, transform_details


def erroneous_registration(source_path: str,
                           output_path: str,
                           atlas_image_path: str,
                           transform_debug_dir: str = None,
                           random_seed: int = None) -> None:
    fixed_image = ants.image_read(atlas_image_path)
    moving_image = ants.image_read(source_path)

    # Perform initial registration
    registration_result = ants.registration(fixed=fixed_image,
                                            moving=moving_image,
                                            type_of_transform='Rigid',
                                            random_seed=random_seed)

    # Extract the affine registration matrix
    registration_transform = ants.read_transform(
        registration_result['fwdtransforms'][0])
    registration_matrix = registration_transform.parameters

    # Apply noise to the matrix elements
    noisy_matrix = registration_matrix.copy().reshape(4, 3)

    # Apply Gaussian noise to rotation, shearing, and scaling indices
    noisy_matrix[:3, :] += np.random.normal(0, 0.1, size=(3, 3))

    # Apply uniform noise to translation indices
    noisy_matrix[3, :] += np.random.uniform(-5, 5, size=3)

    # Create a new affine transform with the noisy matrix
    noisy_transform = ants.create_ants_transform(
        transform_type='AffineTransform',
        dimension=3,
        parameters=noisy_matrix.flatten(),
        fixed_parameters=registration_transform.fixed_parameters)

    # Apply the erroneous registration matrix
    transformed_image = noisy_transform.apply_to_image(image=moving_image,
                                                       reference=fixed_image)

    # Save the transformed image
    ants.image_write(transformed_image, output_path)

    if transform_debug_dir is not None:
        # Save the original and erroneous registration transform
        original_transform_path = os.path.join(
            transform_debug_dir,
            os.path.basename(output_path).replace('.nii.gz', '_orig.mat'))
        ants.write_transform(registration_transform, original_transform_path)
        noisy_transform_path = os.path.join(
            transform_debug_dir,
            os.path.basename(output_path).replace('.nii.gz', '_noisy.mat'))
        ants.write_transform(noisy_transform, noisy_transform_path)
