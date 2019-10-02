conda activate tensorflow

for cam in "K1"
do
    for num in 1
    do
        python /home/rriverag/experiments/run.py \
            --model="i3d" \
            --dataset_name="dahlia" \
            --dahlia_modality="rgb" \
            --dahlia_train_path=/data/stars/share/DAHLIA-full/splits/single_view/train_split_${cam}_${num}.txt \
            --dahlia_validation_path=/data/stars/share/DAHLIA-full/splits/single_view/validation_split_${cam}_${num}.txt \
            --dahlia_test_path=/data/stars/share/DAHLIA-full/splits/single_view/test_split_${cam}_${num}.txt \
            --log_dir=/data/stars/user/rriverag/experiments \
            --num_epochs=2 \
            --batch_size=2 \
            --num_gpus=4 \
            --eval_batch_size=2 \
            --experiment_name="dahlia_rgb_${cam}_${num}_nice"
    done
done