import argparse
import os
import pandas as pd


def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        if 'processed_files.csv' in files:
            csv_files.append(os.path.join(root, 'processed_files.csv'))
    return csv_files


def generate_imglist(input_csv_paths, base_dir, output_dir, labels):
    os.makedirs(output_dir, exist_ok=True)

    for input_csv_path in input_csv_paths:
        # Read the CSV file
        df = pd.read_csv(input_csv_path)

        # Define the split mapping
        split_map = {'TRAIN': 'train', 'VALIDATION': 'val', 'TEST': 'test'}

        # Create the label mapping
        label_map = {label: idx for idx, label in enumerate(labels)} \
            if 'Label' in df.columns else {None: 0}

        # Check if 'Split' column exists, otherwise use default_split
        if 'Split' not in df.columns:
            df['Split'] = 'test'
        else:
            df['Split'] = df['Split'].map(split_map)

        relative_images_dir = os.path.relpath(os.path.dirname(input_csv_path),
                                              base_dir)
        imglist_name = relative_images_dir.replace('/', '_')

        # Group by 'Split' and write to respective text files
        for split, group in df.groupby('Split'):
            split_file_path = os.path.join(output_dir,
                                           f'{split}_{imglist_name}.txt')
            with open(split_file_path, 'w') as f:
                for _, row in group.iterrows():
                    relative_output = os.path.relpath(row['Output'], base_dir)
                    label = label_map[row.get('Label', None)]
                    f.write(f'{relative_output} {label}\n')
            print(f"Image list '{split}_{imglist_name}.txt'"
                  f' (#items={len(group)}) was generated successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate image list text files from CSV.')
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,
                        help='Directory containing processed_files.csv file.')
    parser.add_argument('--base_dir',
                        type=str,
                        required=True,
                        help='Base directory.')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Output directory.')
    parser.add_argument('--labels',
                        type=str,
                        nargs='+',
                        required=False,
                        help='List of labels.')

    args = parser.parse_args()

    csv_files = find_csv_files(args.input_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"'processed_files.csv' not found in {args.input_dir}"
            f' or its subdirectories.')

    generate_imglist(csv_files, args.base_dir, args.output_dir, args.labels)
