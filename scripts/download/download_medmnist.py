import argparse

import medmnist
import os
import pandas as pd


def convert_to_imglists(img_folder, csv_path):
    df = pd.read_csv(csv_path,
                     header=None,
                     usecols=[0, 1, 2],
                     names=['split', 'filename', 'label'])
    split_map = {'TRAIN': 'train', 'VALIDATION': 'val', 'TEST': 'test'}
    imglists = {'train': [], 'val': [], 'test': []}
    for idx, row in df.iterrows():
        image_path = f"{img_folder}/{row['filename']}"
        imglists[split_map[row['split']]].append(
            f"{image_path} {row['label']}")
    return imglists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download MedMNIST datasets')
    parser.add_argument('--datasets', nargs='+', default=['all'])
    parser.add_argument('--save_dir', default='./data/medmnist')
    parser.add_argument('--imglist_dir',
                        default='./data/benchmark_imglist/medmnist')
    parser.add_argument('--size',
                        type=int,
                        default=28,
                        choices=[28, 64, 128, 224])
    args = parser.parse_args()

    if args.datasets[0] == 'all':
        args.datasets = medmnist.INFO.keys()
    os.makedirs(args.imglist_dir, exist_ok=True)

    postfix = f'_{args.size}' if args.size != 28 else ''
    for flag in args.datasets:
        ext = 'gif' if flag.endswith('3d') else 'png'
        flag_xt = f'{flag}{postfix}'
        folder = f'{args.save_dir}/{flag}{postfix}'
        os.makedirs(folder, exist_ok=True)
        csv_path = f'{folder}/{flag_xt}.csv'
        # dataset.save append to csv file, so remove it first
        if os.path.exists(csv_path):
            os.remove(csv_path)
        for split in ['train', 'val', 'test']:
            print(f'Saving {flag_xt} {split}...')
            dataset = getattr(medmnist, medmnist.INFO[flag]['python_class'])(
                split=split, download=True, root=folder, size=args.size)
            dataset.save(folder, ext)
        imglists = convert_to_imglists(f'{flag_xt}/{flag_xt}', csv_path)
        for split, imglist in imglists.items():
            with open(f'{args.imglist_dir}/{split}_{flag_xt}.txt', 'w') as f:
                f.write('\n'.join(imglist) + '\n')
