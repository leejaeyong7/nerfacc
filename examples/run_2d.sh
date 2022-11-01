SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python train_2d_freq_mlp_nerf.py --data_path=/mnt/data/nerf_synthetic --log_path=logs --run_name=freq_train_2d_$SCENE --scene=$SCENE --train_split=train
done
