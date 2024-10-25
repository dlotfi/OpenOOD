import os
import random
from functools import wraps
from typing import Callable

import SimpleITK as sitk
import ants
import numpy as np
import torchio as tio

TransformFunc = Callable[[sitk.Image], sitk.Image]


def sitk_to_torchio(func):
    @wraps(func)
    def wrapper(image: sitk.Image, *args, **kwargs) -> sitk.Image:
        torchio_image = tio.Image.from_sitk(image)
        torchio_image = func(torchio_image, *args, **kwargs)
        return torchio_image.as_sitk()

    return wrapper


@sitk_to_torchio
def motion_artifact(image: tio.Image) -> tio.Image:
    motion_transform = tio.transforms.RandomMotion(
        degrees=10,  # Rotation range [-10, 10] degrees
        translation=10  # Translation range [-10, 10] mm
    )
    return motion_transform(image)


@sitk_to_torchio
def ghost_artifact(image: tio.Image) -> tio.Image:
    ghost_transform = tio.transforms.RandomGhosting(
        num_ghosts=(2, 5),  # Number of ghosted copies
        axes=(0, 1, 2),  # Axes along which to apply the ghosting
        intensity=(0.5, 1)  # Intensity range of the ghosting effect
    )
    return ghost_transform(image)


@sitk_to_torchio
def bias_artifact(image: tio.Image) -> tio.Image:
    bias_transform = tio.transforms.RandomBiasField(
        coefficients=0.5  # Coefficient for the polynomial basis functions
    )
    return bias_transform(image)


@sitk_to_torchio
def spike_artifact(image: tio.Image) -> tio.Image:
    spike_transform = tio.transforms.RandomSpike(
        num_spikes=1,  # Number of spikes to add
        intensity=(0.1, 1)  # Intensity range of the spikes
    )
    return spike_transform(image)


@sitk_to_torchio
def gaussian_noise(image: tio.Image) -> tio.Image:
    # Convert image to NumPy array
    image_data = image.numpy()
    # Normalize the image
    image_data = (image_data - image_data.mean()) / image_data.std()
    # Convert back to TorchIO image
    image = tio.Image(tensor=image_data, affine=image.affine)
    # Add Gaussian noise
    noise_transform = tio.transforms.RandomNoise(
        mean=0,  # Mean of the Gaussian noise
        std=0.5  # Standard deviation of the Gaussian noise
    )
    return noise_transform(image)


@sitk_to_torchio
def downsampling(image: tio.Image) -> tio.Image:
    # Define the downsampling factors for each axis
    downsample_factors = (1, 1, 3)  # Example: downsample along the z-axis
    # Downsample the image
    downsampled_image = tio.transforms.Resample(
        target=(image.spacing[0] * downsample_factors[0],
                image.spacing[1] * downsample_factors[1],
                image.spacing[2] * downsample_factors[2]))(image)
    # Resample back to the original resolution
    resampled_image = tio.transforms.Resample(
        target=image.spacing)(downsampled_image)
    return resampled_image


@sitk_to_torchio
def scaling_perturbation(image: tio.Image) -> tio.Image:
    if random.random() < 0.5:
        # Double the size
        scales = (2, 2, 2)
    else:
        # Shrink by a factor of half
        scales = (0.5, 0.5, 0.5)
    scaling_transform = tio.transforms.RandomAffine(
        scales=scales,
        degrees=0  # No rotation
    )
    return scaling_transform(image)


@sitk_to_torchio
def gamma_alteration(image: tio.Image) -> tio.Image:
    gamma = random.choice([4.5, -4.5])
    gamma_transform = tio.transforms.Lambda(lambda x: x**gamma,
                                            types_to_apply=[tio.INTENSITY])
    return gamma_transform(image)


@sitk_to_torchio
def truncation(image: tio.Image) -> tio.Image:
    # Get the original size of the image
    original_size = np.array(image.shape)

    # Randomly choose a direction (axis) to truncate
    axis = random.choice([0, 1, 2])

    # Calculate the new size after truncation
    new_size = original_size.copy()
    new_size[axis] = new_size[axis] // 2

    # Randomly choose the start index for truncation
    start_index = random.randint(0, original_size[axis] - new_size[axis])

    # Create slices for truncation
    slices = [slice(None)] * 3
    slices[axis] = slice(start_index, start_index + new_size[axis])

    # Apply truncation
    truncated_image = image.data[slices]

    # Create a new TorchIO image with the truncated data
    truncated_image = tio.Image(tensor=truncated_image, affine=image.affine)

    return truncated_image


def erroneous_registration(source_path: str,
                           output_path: str,
                           atlas_image_path: str,
                           transform_debug_dir: str = None) -> None:
    fixed_image = ants.image_read(atlas_image_path)
    moving_image = ants.image_read(source_path)

    # Perform initial registration
    registration_result = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='Affine',
    )

    # Extract the affine registration matrix
    registration_matrix = ants.read_transform(
        registration_result['fwdtransforms'][0])
    matrix = registration_matrix.parameters

    # Apply noise to the matrix elements
    noisy_matrix = np.array(matrix).reshape(3, 4)

    # Apply Gaussian noise to rotation, shearing, and scaling indices
    noisy_matrix[:3, :3] += np.random.normal(0, 0.1, size=(3, 3))

    # Apply uniform noise to translation indices
    noisy_matrix[:, 3] += np.random.uniform(-5, 5, size=3)

    # Create a new affine transform with the noisy matrix
    noisy_transform = ants.create_ants_transform(
        transform_type='AffineTransform',
        dimension=3,
        parameters=noisy_matrix.flatten().tolist(),
        fixed_parameters=registration_matrix.fixed_parameters)

    # Apply the erroneous registration matrix
    transformed_image = ants.apply_transforms(fixed=fixed_image,
                                              moving=moving_image,
                                              transformlist=[noisy_transform])

    # Save the transformed image
    ants.image_write(transformed_image, output_path)

    if transform_debug_dir is not None:
        # Save the erroneous registration transform
        noisy_transform_path = os.path.join(
            transform_debug_dir,
            os.path.basename(output_path).replace('.nii.gz', '.mat'))
        ants.write_transform(noisy_transform, noisy_transform_path)
