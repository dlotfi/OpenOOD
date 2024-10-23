import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class ATLAS2_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self, split: str = None) -> List[FilePair]:
        pattern = re.compile(
            r'(Training|Testing)/R\d{3}/(sub-r\d{3}s\d{3})/ses-1/anat/'
            r'\2_ses-1_space-MNI152NLin2009aSym_T1w\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        # Filter files based on the split type
        if split is not None:
            split_name = 'Training' if split == 'Train' else 'Testing'
            candidate_files = [
                f for f in candidate_files if f.Match.group(1) == split_name
            ]
            self.logger.info(f'Total {len(candidate_files)} files are'
                             f' in the specified split: {split}.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file, match in candidate_files:
            split_part = match.group(1)[:-3]  # Remove the last 'ing' part
            sub_part = match.group(2)
            output_name = f'ATLAS2_{split_part}_{sub_part}_T1.nii.gz'
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing ATLAS2 dataset')
        self.logger.info(self.cfg)
        sampled_files = self.find_and_sample_files(split='Train')
        processed_files = self.process_brain_mris(sampled_files)
        # todo: Questions:
        # 1. Normalize the images?
        # 2. Sample from Train, Test or both?
        self.log_processed_files(processed_files)
        self.logger.info('ATLAS2 dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = ATLAS2_PreProcessor(cfg)
    preprocessor.run()
