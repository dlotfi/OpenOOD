# sh ./scripts/download/dowanload_medmnist_plus.sh

python ./scripts/download/download_medmnist.py \
	--datasets 'organamnist' 'organcmnist' 'organsmnist' 'pneumoniamnist' \
             'chestmnist' 'dermamnist' 'retinamnist' 'breastmnist' \
             'bloodmnist' 'pathmnist'  \
	--save_dir './data/medmnist_plus' \
	--imglist_dir './data/benchmark_imglist/medmnist_plus' \
	--size 224
