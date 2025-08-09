export PYTHONPATH=/data2/lbliao/Code/nnUNet:$PYTHONPATH
#cd ../../
#python x2nnUnet.py
export nnUNet_raw="/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_raw/"
export nnUNet_preprocessed="/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/NAS3/lbliao/Data/MXB/segment/dataset/nnUnet/nnUNet_trained_models"
#nnUNetv2_plan_and_preprocess -d 007  --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train "Dataset007_GLAND" "2d" 0
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train "Dataset005_GLAND" "2d" 1
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train "Dataset005_GLAND" "2d" 2
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train "Dataset005_GLAND" "2d" 3
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_train "Dataset005_GLAND" "2d" 4
#nnUNetv2_predict -i '/NAS3/lbliao/Data/MXB/seminal/dataset/nnUnet/nnUNet_raw/Dataset001_SE512/imagesTr' -o '/NAS3/lbliao/Data/MXB/seminal/dataset/nnUnet/nnUNet_raw/Dataset001_SE512/result' -d 'Dataset001_SE512' -c '2d'