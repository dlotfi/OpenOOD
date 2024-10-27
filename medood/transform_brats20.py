import os
from typing import List

import pandas as pd
import SimpleITK as sitk

from preprocessor_base import BaseBrainPreProcessor, FilePair, ProcessFunc
from preprocessor_config import PreProcessorBrainConfig
from utils import random_sample
from transformations import (motion_artifact, ghost_artifact, bias_artifact,
                             spike_artifact, gaussian_noise, downsampling,
                             scaling_perturbation, gamma_alteration,
                             truncation, erroneous_registration, TransformFunc)


class BraTS20_Transformer(BaseBrainPreProcessor):
    def transform_images(self, source_output_pairs: List[FilePair],
                         transform_name: str,
                         transform_func: ProcessFunc) -> List[FilePair]:

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

        df_filtered = df[df[['Split']] == 'TEST']

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

    def _wrap_transform(self, transform: TransformFunc) -> ProcessFunc:
        def wrapped_transform(pair: FilePair) -> None:
            image = sitk.ReadImage(pair.Source)
            image = transform(image)
            image = self._normalize_image(image)
            sitk.WriteImage(image, pair.Output)

        return wrapped_transform

    def _erroneous_registration(self, pair: FilePair) -> None:
        transforms_debug_dir = os.path.join(self.cfg.output_dir,
                                            'registration/transforms')
        erroneous_registration(pair.Source, pair.Output,
                               self._atlas_image_path, transforms_debug_dir)
        image = sitk.ReadImage(pair.Output)
        image = self._normalize_image(image)
        sitk.WriteImage(image, pair.Output)

    def run(self):
        all_transformations = {
            'Motion': self._wrap_transform(motion_artifact),
            'Ghost': self._wrap_transform(ghost_artifact),
            'Bias': self._wrap_transform(bias_artifact),
            'Spike': self._wrap_transform(spike_artifact),
            'Gaussian': self._wrap_transform(gaussian_noise),
            'Downsampling': self._wrap_transform(downsampling),
            'Scaling': self._wrap_transform(scaling_perturbation),
            'Gamma': self._wrap_transform(gamma_alteration),
            'Truncation': self._wrap_transform(truncation),
            'Registration': self._erroneous_registration
        }

        self.logger.info('Start synthesizing transformed BraTS2020 dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in 'Test' split of the pre-processed BraTS2020
        #    and sample randomly from them
        sampled_files = self.find_and_sample_files()
        # 2. Apply various transformations to all T1 sampled images
        for name, transform in all_transformations.items():
            os.makedirs(os.path.join(self.cfg.output_dir, name.lower()),
                        exist_ok=True)
            transform_sampled_files = [
                FilePair(f.Source, f.Output.format(name.lower()))
                for f in sampled_files
            ]
            processed_files = self.transform_images(transform_sampled_files,
                                                    name, transform)
            self.save_processed_files(processed_files)
        self.logger.info('BraTS20 dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    transformer = BraTS20_Transformer(cfg)
    transformer.run()
