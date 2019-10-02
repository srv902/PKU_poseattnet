conda activate tensorflow

python /home/rriverag/experiments/run.py \
    --dataset_name="pku-mmd" \
    --model="lstm" \
    --pku_mmd_modality="skeletons" \
    --pku_mmd_split="cross-view" \
    --pku_mmd_rgb_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Data/RGB_VIDEO \
    --pku_mmd_skeletons_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Data/PKU_Skeleton_Renew \
    --pku_mmd_labels_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Label/Train_Label_PKU_final \
    --pku_mmd_splits_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Split\
    --log_dir=/data/stars/user/rriverag/experiments \
    --num_epochs=100 \
    --batch_size=32 \
    --eval_batch_size=128 \
    --data_size=150 \
    --experiment_name="pku_skeletons"