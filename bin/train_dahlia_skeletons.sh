conda activate tensorflow

for cam in "K1"
do
    for num in 1
    do
        python /home/rriverag/experiments/run.py \
            --model="lstm" \
            --dataset_name="dahlia" \
            --dahlia_modality="skeletons" \
            --dahlia_train_path=/data/stars/share/DAHLIA-full/splits/single_view/train_split_${cam}_${num}.txt \
            --dahlia_validation_path=/data/stars/share/DAHLIA-full/splits/single_view/validation_split_${cam}_${num}.txt \
            --dahlia_test_path=/data/stars/share/DAHLIA-full/splits/single_view/test_split_${cam}_${num}.txt \
            --dahlia_skeletons_dir=/data/stars/share/DAHLIA-full/skeletons/proc \
            --log_dir=/data/stars/user/rriverag/experiments \
            --num_epochs=100 \
            --num_gpus=4 \
            --batch_size=32 \
            --eval_batch_size=128 \
            --data_size=75 \
            --experiment_name="dahlia_skeletons_${cam}_${num}"
    done
done