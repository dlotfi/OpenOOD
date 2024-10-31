import os
import re
from typing import List

import pandas as pd
import pydicom

from preprocessor_base import (BaseDICOMPreProcessor, BaseBrainPreProcessor,
                               FilePair, TempFilePair, Window)
from preprocessor_config import PreProcessorBrainConfig
from utils import random_sample, insert_subdir, remove_subdir


class CQ500_PreProcessor(BaseDICOMPreProcessor, BaseBrainPreProcessor):
    def find_and_sample_dicom_series(self) -> List[FilePair]:
        csv_path = os.path.join(self.cfg.base_dir, '../reads.csv')
        df = pd.read_csv(csv_path)

        # Filter rows where at least two out of three radiologists' reads are
        # positive regarding intracranial hemorrhage or cranial fractures
        # df_filtered = df[(df[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(1) >= 2) | (
        #     df[['R1:Fracture', 'R2:Fracture', 'R3:Fracture']].sum(1) >= 2)]

        # Extract patient ids from the 'name' column and sort them
        df['name'] = df['name'].str.replace('CQ500-CT-', '').astype(int)
        patient_ids = sorted(df['name'].tolist())

        pre_contrast_pattern = re.compile(r'PLAIN|PRE CONTRAST|0.625mm',
                                          re.IGNORECASE)

        candidate_dicom_series = []

        for patient_id in patient_ids:
            patient_folder = os.path.join(
                self.cfg.base_dir, f'CQ500CT{patient_id} CQ500CT{patient_id}')

            if not os.path.exists(patient_folder):
                self.logger.warning(
                    f'Patient folder not found: {patient_folder}')
                continue

            found_series = []
            for root, _, files in os.walk(patient_folder):
                if 'Unknown Study' not in root:
                    continue
                series_name = os.path.basename(root)
                dicom_files = [
                    os.path.join(root, f) for f in files
                    if f.lower().endswith('.dcm')
                ]

                # Exclude series with fewer than 200 slices, as they likely
                # contain only a partial view of the brain, or with more than
                # 400 slices, as they likely include other body parts.
                if len(dicom_files) < 200 or len(dicom_files) > 400:
                    continue

                # Filter series based on slice thickness and ensure that the
                # series name indicates it is pre-contrast. Exclude any series
                # with 'BODY' in their FilterType, as they are most likely not
                # brain scans. Additionally, consider only series labeled as
                # 'PRIMARY'.
                ds = pydicom.dcmread(dicom_files[0])
                if float(ds.SliceThickness) <= 0.625 and \
                   'BODY' not in ds.FilterType and \
                   'PRIMARY' in ds.ImageType and \
                   pre_contrast_pattern.search(series_name):
                    found_series.append((root, len(dicom_files)))

            if found_series:
                # find the series with the most number of slices and
                # the shortest series name
                found_series = sorted(found_series,
                                      key=lambda x: (-x[1], len(x[0])))
                best_series_path = found_series[0][0]
                output_nifti = os.path.join(self.cfg.output_dir,
                                            f'CQ500_{patient_id}_CT.nii.gz')
                candidate_dicom_series.append(
                    FilePair(best_series_path, output_nifti))
                self.logger.info(
                    f'Found {len(found_series)} series matching the'
                    f' criteria for patient {patient_id}: '
                    f" {', '.join (str(f) for f in found_series)}")
            else:
                self.logger.warning(f'No series matching the criteria'
                                    f' found for patient {patient_id}.')

        self.logger.info(f'Found total {len(candidate_dicom_series)} DICOM'
                         f' series matching the required criteria.')

        sampled_dicom_series = random_sample(candidate_dicom_series,
                                             self.cfg.num_samples)
        self.logger.info(f'Sampled {len(sampled_dicom_series)} DICOM series.')

        return sampled_dicom_series

    def run(self):
        self.logger.info('Start preprocessing CQ500 dataset')
        self.logger.info(self.cfg)
        subdir = 'raw_nifti'
        # 1. Find all DICOM series corresponding to the pre-contrast CTs
        #    having thin slices and, containing the whole brain and sample
        #    randomly from them
        sampled_dicom_series = self.find_and_sample_dicom_series()
        # 2. Convert all sampled series to NIfTI while applying windowing
        os.makedirs(os.path.join(self.cfg.output_dir, subdir), exist_ok=True)
        sampled_dicom_series = [
            FilePair(f.Source, insert_subdir(f.Output, subdir))
            for f in sampled_dicom_series
        ]
        converted_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series, apply_window=Window(40, 100))
        # 3. Register to SRI24, skull-strip, and normalize all sampled images
        files_to_be_processed = [
            TempFilePair(Source=f.Output,
                         Output=remove_subdir(f.Output, subdir),
                         OriginalSource=f.Source) for f in converted_files
        ]
        processed_files = self.register_skullstrip_normalize_images(
            files_to_be_processed)
        self.save_processed_files(
            [FilePair(f.OriginalSource, f.Output) for f in processed_files])
        self.logger.info('CQ500 dataset preprocessing completed.')
        # Questions:
        # * Registration and skull-stripping?
        # * Apply windowing?
        # * Which folder?
        # * Filter on ICH and Fracture?


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = CQ500_PreProcessor(cfg)
    preprocessor.run()
