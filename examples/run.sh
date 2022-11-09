SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_1d_$SCENE --scene=$SCENE --train_split=train --model_type=1d
done

SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_2d_$SCENE --scene=$SCENE --train_split=train --model_type=2d
done
