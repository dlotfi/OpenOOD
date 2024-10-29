import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class WMH2017_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self, split: str = None) -> List[FilePair]:
        pattern = re.compile(
            r'(training|test)/'
            r'(?:Amsterdam/(?:GE3T|GE1T5|Philips_VU\s\.PETMR_01\.)|'
            r'Singapore|Utrecht)/'
            r'(\d+)/pre/3DT1\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        # Filter files based on the split type
        if split is not None:
            split_name = 'training' if split == 'Train' else 'test'
            candidate_files = [
                f for f in candidate_files if f.Match.group(1) == split_name
            ]
            self.logger.info(f'Total {len(candidate_files)} files'
                             f' are in the specified split: {split}.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file in candidate_files:
            split_part = \
                'Train' if file.Match.group(1) == 'training' else 'Test'
            number = file.Match.group(2)
            output_name = f'WMH2017_{split_part}_{number}_T1.nii.gz'
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file.FilePath, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing WMH2017 dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in  both 'Train' and 'Test' splits
        #    and sample randomly from them
        sampled_files = self.find_and_sample_files()
        # 2. Register to SRI24, skull-strip, and normalize all sampled images
        processed_files = self.register_skullstrip_normalize_images(
            sampled_files)
        self.save_processed_files(processed_files)
        self.logger.info('WMH2017 dataset preprocessing completed.')
        # Questions:
        # * Which folder: 'orig' or 'pre'?
        # * Which file: 3DT1 or T1?
        # * Include 'GE1T5' and 'Philips_VU .PETMR_01'?


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = WMH2017_PreProcessor(cfg)
    preprocessor.run()
