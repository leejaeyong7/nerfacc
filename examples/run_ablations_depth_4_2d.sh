SCENE_IDS=(mic lego chair drums ficus hotdog materials ship)
DEPTH=4
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_2d_l7_f16_d${DEPTH}_${SCENE} --scene=${SCENE} --train_split=train --model_type=2d --net_depth=$DEPTH
done

