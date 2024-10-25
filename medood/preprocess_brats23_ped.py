import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class BraTS23_PED_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self, split: str = None) -> List[FilePair]:
        pattern = re.compile(
            r'(ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData'
            r'|ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData)'
            r'/BraTS-PED-(\d{5})-000/BraTS-PED-\2-000-t1n\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        # Filter files based on the split type
        if split is not None:
            split_name = 'TrainingData' if split == 'Train' \
                else 'ValidationData'
            candidate_files = [
                f for f in candidate_files if split_name in f.Match.group(1)
            ]
            self.logger.info(f'Total {len(candidate_files)} files are'
                             f' in the specified split: {split}.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file in candidate_files:
            split_part = \
                'Train' if 'TrainingData' in file.Match.group(1) \
                else 'Validation'
            number = file.Match.group(2)
            output_name = f'BraTS23_PED_{split_part}_{number}_T1.nii.gz'
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file.FilePath, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing BraTS2023'
                         ' Pediatric Tumors dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in 'Train' split and sample randomly from them
        sampled_files = self.find_and_sample_files(split='Train')
        # 2. Normalize all sampled images
        processed_files = self.normalize_images(sampled_files)
        self.save_processed_files(processed_files)
        self.logger.info('BraTS2023 Pediatric Tumors dataset'
                         ' preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = BraTS23_PED_PreProcessor(cfg)
    preprocessor.run()
