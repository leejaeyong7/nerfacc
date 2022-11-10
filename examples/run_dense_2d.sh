SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs_dense.py --data_path=data/nerf_synthetic --run_name=freq_2d_l7_f16_d2_$SCENE --scene=$SCENE --train_split=train --model_type=2d
done
