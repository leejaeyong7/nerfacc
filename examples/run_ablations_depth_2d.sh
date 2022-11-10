SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
DEPTHS=(3 4)
for SCENE in ${SCENE_IDS[@]}; do
    for DEPTH in ${DEPTHS[@]}; do
        python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_2d_l7_f16_d${DEPTH}_${SCENE} --scene=${SCENE} --train_split=train --model_type=2d --net_depth=$DEPTH
    done
done

