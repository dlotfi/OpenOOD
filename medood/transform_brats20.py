import json
import os
from typing import List

import pandas as pd
import SimpleITK as sitk

from preprocessor_base import (BaseBrainPreProcessor, FilePair, ProcessFunc,
                               T_FilePair)
from preprocessor_config import PreProcessorBrainConfig
from utils import random_sample
from transformations import (motion_artifact, ghost_artifact, bias_artifact,
                             spike_artifact, gaussian_noise, downsampling,
                             scaling_perturbation, gamma_alteration,
                             truncation, erroneous_registration)


class BraTS20_Transformer(BaseBrainPreProcessor):
    _all_transformations = {
        'Motion': motion_artifact,
        'Ghost': ghost_artifact,
        'Bias': bias_artifact,
        'Spike': spike_artifact,
        'Noise': gaussian_noise,
        'Downsampling': downsampling,
        'Scaling': scaling_perturbation,
        'Gamma': gamma_alteration,
        'Truncation': truncation,
        'Registration': erroneous_registration
    }

    def transform_images(self, source_output_pairs: List[T_FilePair],
                         transform_name: str,
                         transform_func: ProcessFunc) -> List[T_FilePair]:

        self.logger.info(f"Applying '{transform_name}' transform"
                         f' to brain volumes')
        processed_pairs = [
            self._process_image(pair, transform_func)
            for pair in source_output_pairs
        ]
        processed_pairs = [
            pair for pair in processed_pairs if pair is not None
        ]
        self.logger.info(f'{len(processed_pairs)} volumes have been'
                         f' transformed.')
        return processed_pairs

    def find_and_sample_files(self) -> List[FilePair]:
        csv_path = os.path.join(self.cfg.base_dir, 'processed_files.csv')
        df = pd.read_csv(csv_path)

        df_filtered = df[df['Split'] == 'TEST']

        candidate_files = []
        for _, row in df_filtered.iterrows():
            input_path = row['Source']
            output_file_name = str(
                os.path.basename(row['Output']).replace('T1', 'T1_{0}'))
            output_path = os.path.join(self.cfg.output_dir, '{0}',
                                       output_file_name)
            candidate_files.append(FilePair(input_path, output_path))

        self.logger.info(f'Found total {len(candidate_files)} files.')

        sampled_files = random_sample(candidate_files, self.cfg.num_samples)

        self.logger.info(f'Sampled {len(sampled_files)} files.')

        return sampled_files

    def _get_transform_func(self, transform_name: str,
                            debug_dir: str) -> ProcessFunc:
        transform_func = self._all_transformations[transform_name]

        def wrapped_erroneous_registration(pair: FilePair) -> None:
            erroneous_registration(pair.Source, pair.Output,
                                   self._atlas_image_path, debug_dir,
                                   self.cfg.seed)
            image = sitk.ReadImage(pair.Output)
            image = self._normalize_image(image)
            sitk.WriteImage(image, pair.Output)

        def wrapped_transform(pair: FilePair) -> None:
            image = sitk.ReadImage(pair.Source)
            image, transform_details = transform_func(image)
            if debug_dir and transform_details:
                transform_details_path = os.path.join(
                    debug_dir,
                    os.path.basename(pair.Output).replace('.nii.gz', '.json'))
                with open(transform_details_path, 'w') as f:
                    json.dump(transform_details, f, indent=4)
            image = self._normalize_image(image)
            sitk.WriteImage(image, pair.Output)

        if transform_name == 'Registration':
            return wrapped_erroneous_registration
        return wrapped_transform

    def run(self):
        self.logger.info('Start synthesizing transformed BraTS2020 dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in 'Test' split of the pre-processed BraTS2020
        #    and sample randomly from them
        sampled_files = self.find_and_sample_files()
        # 2. Apply various transformations to all T1 sampled images
        for name in self._all_transformations.keys():
            output_dir = os.path.join(self.cfg.output_dir, name.lower())
            debug_dir = os.path.join(output_dir, 'transforms')
            os.makedirs(debug_dir, exist_ok=True)
            transform_sampled_files = [
                FilePair(f.Source, f.Output.format(name.lower()))
                for f in sampled_files
            ]
            transform_func = self._get_transform_func(name, debug_dir)
            processed_files = self.transform_images(transform_sampled_files,
                                                    name, transform_func)
            csv_path = os.path.join(output_dir, 'processed_files.csv')
            self.save_processed_files(processed_files, csv_path)
        self.logger.info('BraTS20 dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    transformer = BraTS20_Transformer(cfg)
    transformer.run()
