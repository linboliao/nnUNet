export PYTHONPATH=/data2/lbliao/Code/nnUNet:$PYTHONPATH
export nnUNet_compile=0

export nnUNet_raw="/NAS145/Data/GlandSeg/nnUnet/raw/"
export nnUNet_preprocessed="/NAS145/Data/GlandSeg/nnUnet/preprocessed"
export nnUNet_results="/NAS145/Data/GlandSeg/nnUnet/trained_models"
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train "Dataset003_maixin" "2d" 0

#nnUNetv2_predict -i '/NAS145/Data/GlandSeg/nnUnet/raw/Dataset002_CRAG/imagesTs' -o '/NAS145/Data/GlandSeg/nnUnet/raw/Dataset002_CRAG/predTs' -d 'Dataset002_CRAG' -c '2d' -f 0 -tr nnUNetTrainer