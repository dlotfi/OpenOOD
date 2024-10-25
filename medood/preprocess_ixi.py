import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class IXI_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self) -> List[FilePair]:
        pattern = re.compile(r'IXI\d{3}-(HH|Guys|IOP)-\d{4}-T1\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file in candidate_files:
            output_name = os.path.basename(file.FilePath)
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file.FilePath, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing IXI dataset')
        self.logger.info(self.cfg)
        # 1. Find all files and sample randomly from them
        sampled_files = self.find_and_sample_files()
        # 2. Register to SRI24, skull-strip, and normalize all sampled images
        processed_files = self.register_skullstrip_normalize_images(
            sampled_files)
        self.save_processed_files(processed_files)
        self.logger.info('IXI dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = IXI_PreProcessor(cfg)
    preprocessor.run()
