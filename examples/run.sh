python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_1d_l5_f16_d8_lego --scene=lego --train_split=train --model_type=1d --log2_res=5



# for SCENE in ${SCENE_IDS[@]}; do
#     python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_1d_$SCENE --scene=$SCENE --train_split=train --model_type=1d
# done

