# conda activate tensorflow

python ../run.py \
    --dataset_name="pku-mmd" \
    --model="i3d" \
    --pku_mmd_modality="rgb" \
    --pku_mmd_split="cross-subject" \
    --pku_mmd_rgb_dir=/data/stars/share/PKU-MMD/cropped-person \
    --pku_mmd_labels_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Label/Train_Label_PKU_final \
    --pku_mmd_splits_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Split\
    --log_dir=/data/stars/user/rriverag/experiments \
    --num_epochs=2 \
    --batch_size=4 \
    --eval_batch_size=4 \
    --num_gpus=4 \
    --gpus="True" \
    --class_weight="True" \
    --experiment_name="pku_rgb_cross_subject"
