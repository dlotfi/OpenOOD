# sh ./scripts/download/dowanload_medmnist.sh

python ./scripts/download/download_medmnist.py \
	--datasets 'organamnist' 'organcmnist' 'organsmnist' 'pneumoniamnist' \
             'chestmnist' 'dermamnist' 'retinamnist' 'breastmnist' \
             'bloodmnist' 'tissuemnist' 'pathmnist'  \
	--save_dir './data/medmnist' \
	--imglist_dir './data/benchmark_imglist/medmnist' \
	--size 28

#python ./scripts/download/download_medmnist.py \
#	--datasets 'organmnist3d' 'nodulemnist3d' 'fracturemnist3d' \
#	           'adrenalmnist3d' 'vesselmnist3d' 'synapsemnist3d' \
#	--save_dir './data/medmnist3d' \
#	--imglist_dir './data/benchmark_imglist/medmnist3d' \
#	--size 64
