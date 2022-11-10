SCENE_IDS=(mic lego chair drums ficus hotdog materials ship)
FEATURE=32
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_1d_l7_f${FEATURE}_d2_$SCENE --scene=$SCENE --train_split=train --model_type=1d --num_f=$FEATURE
done

