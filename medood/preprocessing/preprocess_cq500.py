import os
from typing import List

import pandas as pd
import pydicom

from preprocessor_base import (BaseDICOMPreProcessor, BaseBrainPreProcessor,
                               FilePair)
from preprocessor_config import PreProcessorBrainConfig
from utils import random_sample


class CQ500_PreProcessor(BaseDICOMPreProcessor, BaseBrainPreProcessor):
    def find_and_sample_dicom_series(self) -> List[FilePair]:
        csv_path = os.path.join(self.cfg.base_dir, '../reads.csv')
        df = pd.read_csv(csv_path)

        # Filter rows where at least two out of three columns are 1
        df_filtered = df[(df[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(1) >= 2) | (
            df[['R1:Fracture', 'R2:Fracture', 'R3:Fracture']].sum(1) >= 2)]

        candidate_dicom_series = []

        for _, row in df_filtered.iterrows():
            patient_id = row['name'].replace('CQ500-CT-', '')
            patient_folder = os.path.join(
                self.cfg.base_dir, f'CQ500CT{patient_id} CQ500CT{patient_id}')

            if not os.path.exists(patient_folder):
                self.logger.warning(
                    f'Patient folder not found: {patient_folder}')
                continue

            found_series = 0
            for root, _, files in os.walk(patient_folder):
                if 'Unknown Study' not in root:
                    continue
                series_name = os.path.basename(root)
                dicom_files = [
                    os.path.join(root, f) for f in files
                    if f.lower().endswith('.dcm')
                ]

                # Read metadata of the first DICOM file in the series
                slice_thickness = 0
                if dicom_files:
                    ds = pydicom.dcmread(dicom_files[0])
                    slice_thickness = float(ds.SliceThickness)

                if slice_thickness == 0.625 and 'plain' in series_name.lower():
                    output_nifti = os.path.join(
                        self.cfg.output_dir, f'CQ500_{patient_id}_CT.nii.gz')
                    candidate_dicom_series.append(FilePair(root, output_nifti))
                    found_series += 1

            if found_series == 0:
                self.logger.warning(f'No series matching the criteria'
                                    f' found for patient {patient_id}.')
            elif found_series > 1:
                self.logger.warning(f'Found multiple series matching the'
                                    f' criteria for patient {patient_id}.')

        self.logger.info(f'Found total {len(candidate_dicom_series)} DICOM'
                         f' series matching the required criteria.')

        sampled_dicom_series = random_sample(candidate_dicom_series,
                                             self.cfg.num_samples)
        self.logger.info(f'Sampled {len(sampled_dicom_series)} DICOM series.')

        return sampled_dicom_series

    def run(self):
        self.logger.info('Start preprocessing CQ500 dataset')
        self.logger.info(self.cfg)
        # 1. Find all thin plain DICOM series having intracranial
        #    hemorrhage (ICH) cranial fractures and sample randomly from them
        sampled_dicom_series = self.find_and_sample_dicom_series()
        # 2. Convert all sampled series to NIfTI while applying windowing
        processed_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series, apply_windowing=True)
        # 3. Register to SRI24, skull-strip, and normalize all sampled images
        brain_process_file_pairs = [
            FilePair(f.Output, f.Output) for f in processed_files
        ]
        brain_process_file_pairs = self.register_skullstrip_normalize_images(
            brain_process_file_pairs)
        self.save_processed_files(processed_files)
        self.logger.info('CQ500 dataset preprocessing completed.')
        # Questions:
        # * Registration and skull-stripping? How?
        # * Apply windowing?
        # * Which folder?
        # * Filter on ICH and Fracture?


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = CQ500_PreProcessor(cfg)
    preprocessor.run()
