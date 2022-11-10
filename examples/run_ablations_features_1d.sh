SCENE_IDS=(lego)
FEATURES=(8 32)
for SCENE in ${SCENE_IDS[@]}; do
    for FEATURE in ${FEATURES[@]}; do
        python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_1d_l7_f${FEATURE}_d2_$SCENE --scene=$SCENE --train_split=train --model_type=1d --num_f=$FEATURE --net_depth=8
    done
done

