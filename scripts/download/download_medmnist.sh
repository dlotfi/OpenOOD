# sh ./scripts/download/dowanload_medmnist.sh

python ./scripts/download/download_medmnist.py \
	--datasets 'organamnist' 'organcmnist' 'organsmnist' 'pneumoniamnist' \
             'chestmnist' 'dermamnist' 'retinamnist' 'breastmnist' \
             'bloodmnist' 'pathmnist'  \
	--save_dir './data/medmnist' \
	--imglist_dir './data/benchmark_imglist/medmnist' \
	--size 28
