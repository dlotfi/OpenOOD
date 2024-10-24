import os
import random
from dataclasses import dataclass, fields
from datetime import datetime

import logging

from abc import ABC, abstractmethod
from typing import List, Tuple

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom
from auxiliary.normalization.percentile_normalizer import PercentileNormalizer
from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator

from preprocessor_config import PreProcessorConfig, PreProcessorBrainConfig


@dataclass
class FilePair:
    Source: str
    Output: str


class BasePreProcessor(ABC):
    def __init__(self, cfg: PreProcessorConfig):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        self._setup_logger()
        self._normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

    def _setup_logger(self, level=logging.DEBUG):
        preprocessor_name = self.__class__.__name__
        self.logger = logging.getLogger(preprocessor_name)
        self.log_dir = os.path.join(self.cfg.output_dir, 'processing_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, f"{preprocessor_name.replace(' ', '_')}_" +
            datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log'))
        # Create handlers
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.log_file)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d %(levelname).1s   %(message)s',
            datefmt='%H:%M:%S')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # Add handlers to the logger
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(level)

    def save_processed_files(self, processed_files: List[FilePair]) -> None:
        if not processed_files:
            self.logger.warning('No processed files to log.')
            return
        csv_path = os.path.join(self.cfg.output_dir, 'processed_files.csv')
        df = pd.DataFrame(
            processed_files,
            columns=[f.name for f in fields(processed_files[0].__class__)])
        df.to_csv(csv_path, index=False)

    def _normalize_image(self, image: sitk.Image) -> sitk.Image:
        image_array = sitk.GetArrayFromImage(image)
        normalized_array = self._normalizer.normalize(image_array)
        return sitk.GetImageFromArray(normalized_array)

    @abstractmethod
    def run(self):
        pass


class BaseBrainPreProcessor(BasePreProcessor, ABC):
    def __init__(self, cfg: PreProcessorBrainConfig):
        super().__init__(cfg)
        self._registrator = ANTsRegistrator()
        self._brain_extractor = HDBetExtractor()

    def process_brain_images(self,
                             source_output_pairs: List[FilePair],
                             skull_stripping: bool = True) -> List[FilePair]:
        processed_pairs = []
        self.logger.info('Registering, Skull-stripping, and Normalizing MRIs')
        for pair in source_output_pairs:
            if os.path.exists(pair.Output) and self.cfg.skip_existing:
                self.logger.info(
                    f'Skipping re-processing the existing file: {pair.Output}')
                processed_pairs.append(pair)
                continue
            try:
                if skull_stripping:
                    center = Modality(
                        modality_name='T1',
                        input_path=pair.Source,
                        normalized_bet_output_path=pair.Output,
                        atlas_correction=True,
                        normalizer=self._normalizer,
                    )
                else:
                    center = Modality(
                        modality_name='T1',
                        input_path=pair.Source,
                        normalized_skull_output_path=pair.Output,
                        atlas_correction=True,
                        normalizer=self._normalizer,
                    )
                # By default, registers the brains to SRI24 atlas
                preprocessor = Preprocessor(
                    center_modality=center,
                    moving_modalities=[],
                    registrator=self._registrator,
                    brain_extractor=self._brain_extractor,
                    use_gpu=self.cfg.use_gpu,
                )
                preprocessor.run(log_file=self.log_file)
                processed_pairs.append(pair)
                self.logger.info(f"Successfully processed '{pair.Source}'"
                                 f" to '{pair.Output}'")
            except Exception as e:
                self.logger.error(f"Error processing '{pair.Source}': {e}")
        self.logger.info(f'{len(processed_pairs)} MRIs have been'
                         f' registered, skull-stripped, and normalized.')

        return processed_pairs


class BaseDICOMPreProcessor(BasePreProcessor, ABC):
    def _dicom_to_nifti(self, dicom_dir: str, output_nifti: str,
                        apply_windowing: bool, normalize: bool) -> List[str]:
        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        output_niftis = []

        if not series_ids:
            self.logger.error(
                f"No DICOM series found in the directory '{dicom_dir}'.")
            return []

        # Iterate over all series found in the directory
        for series_id in series_ids:
            # Get the file names associated with the current series ID
            dicom_series = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
            reader.SetFileNames(dicom_series)

            # Load the DICOM series into a SimpleITK image
            image = reader.Execute()

            if apply_windowing:
                # Read the first DICOM file to get the metadata
                dicom_file = pydicom.dcmread(dicom_series[0])
                window_center = float(dicom_file.WindowCenter)
                window_width = float(dicom_file.WindowWidth)
                image = sitk.IntensityWindowing(
                    image, window_center - window_width / 2,
                    window_center + window_width / 2)
                self.logger.info(
                    f'Applied windowing ({window_center}, {window_width})'
                    f" to series {series_id} in '{dicom_dir}'.")

            if normalize:
                image = self._normalize_image(image)
                self.logger.info(
                    f"Normalized series {series_id} in '{dicom_dir}'.")

            # Create a NIfTI file name based on the series ID
            if len(series_ids) > 1:
                output_nifti = output_nifti.replace('.nii.gz',
                                                    f'_{series_id}.nii.gz')

            # Write to NIfTI format
            sitk.WriteImage(image, output_nifti)
            self.logger.info(f"Series {series_id} in '{dicom_dir}' converted"
                             f" successfully to '{output_nifti}'")
            output_niftis.append(output_nifti)

        return output_niftis

    def convert_dicom_series_to_nifti(
            self,
            source_dicom_output_nifti_pairs: List[FilePair],
            apply_windowing: bool = False,
            normalize: bool = False) -> List[FilePair]:
        self.logger.info('Converting DICOM series to NIfTI')
        processed_pairs = []
        for pair in source_dicom_output_nifti_pairs:
            try:
                output_nifti_files = self._dicom_to_nifti(
                    pair.Source, pair.Output, apply_windowing, normalize)
                for nifti_file in output_nifti_files:
                    processed_pairs.append(FilePair(pair.Source, nifti_file))
            except Exception as e:
                self.logger.error(
                    f"Error converting '{pair.Source}' to NifTi: {e}")
        self.logger.info(f'{len(processed_pairs)} DICOM series'
                         f' have been converted to NIfTI.')
        return processed_pairs


class BaseNonBrainPreProcessor(BasePreProcessor, ABC):
    def _resample_and_crop(self, input_path: str, output_path: str,
                           target_size: Tuple, target_spacing: Tuple) -> None:
        # Read the input image
        image = sitk.ReadImage(input_path)

        # Resample the image to isotropic 1mm^3
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(
                original_size, original_spacing, target_spacing)
        ]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample.Execute(image)

        # Center crop the image to the target size
        resampled_size = resampled_image.GetSize()
        crop_start = [(rsz - tsz) // 2
                      for rsz, tsz in zip(resampled_size, target_size)]
        crop_end = [start + tsz for start, tsz in zip(crop_start, target_size)]
        cropped_image = resampled_image[crop_start[0]:crop_end[0],
                                        crop_start[1]:crop_end[1],
                                        crop_start[2]:crop_end[2]]

        # Write the output image
        sitk.WriteImage(cropped_image, output_path)
        self.logger.info(f"'{input_path}' resampled and center cropped"
                         f" successfully to '{output_path}'")

    def resample_and_center_crop_files(
        self,
        files: List[str],
        target_size: Tuple = (240, 240, 155),
        target_spacing: Tuple = (1.0, 1.0, 1.0)
    ) -> List[str]:
        self.logger.info(f'Resampling files to {target_spacing}'
                         f' and center cropping to {target_size}')
        resampled_cropped_files = []
        for file_path in files:
            try:
                self._resample_and_crop(file_path, file_path, target_size,
                                        target_spacing)
                resampled_cropped_files.append(file_path)
            except Exception as e:
                self.logger.error(
                    f"Error resampling and center cropping '{file_path}': {e}")
        self.logger.info(f'{len(resampled_cropped_files)} files have been'
                         f' resampled and center cropped.')
        return resampled_cropped_files
