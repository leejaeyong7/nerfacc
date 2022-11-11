SCENE_IDS=(lego)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_1d_r7_f16_d8_no_add_$SCENE --scene=$SCENE --train_split=train --model_type=1d
    python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_2d_r7_f16_d8_no_add_$SCENE --scene=$SCENE --train_split=train --model_type=2d
done


# for SCENE in ${SCENE_IDS[@]}; do
#     python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_1d_$SCENE --scene=$SCENE --train_split=train --model_type=1d
# done

