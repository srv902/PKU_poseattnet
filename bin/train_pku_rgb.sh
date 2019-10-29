# conda activate tensorflow
export PYTHONPATH=/data/stars/user/sasharma/PKU_poseattnet/contrib/i3d

python ../run.py \
    --dataset_name="pku-mmd" \
    --model="lstm" \
    --pku_mmd_modality="rgb" \
    --protocol="CS" \
    --pku_mmd_split="cross-subject" \
    --pku_mmd_rgb_dir=/data/stars/share/PKU-MMD/cropped-person \
    --pku_mmd_labels_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Label/Train_Label_PKU_final \
    --pku_mmd_splits_dir=/data/stars/user/rriverag/pku-mmd/PKUMMD/Split\
    --pku_mmd_skeletons_dir=/data/stars/share/PKU-MMD/skeletons \
    --log_dir=/data/stars/user/sasharma/PKU_poseattnet/output\
    --num_epochs=15 \
    --stack_size=10 \
    --n_neuron=64 \
    --timesteps=10 \
    --batch_size=1 \
    --num_gpus=4 \
    --eval_batch_size=1 \
    --gpus="True" \
    --class_weight="True" \
    --mode="test" \
    --use_predict="True" \
    --use_ntu_weights="False" \
    --feature_extract="False" \
    --experiment_name="pku_rgb_cross_subject_I3D_feature_10frames_LSTM_test"  # GRU32"     # 64I3D_1024feat_max"
