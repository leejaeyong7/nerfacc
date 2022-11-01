SCENE_IDS=(chair ficus hotdog lego materials mic ship drums)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freq_mlp_nerf.py --data_path=/mnt/data/nerf_synthetic --log_path=logs --run_name=freq_train_$SCENE --scene=$SCENE --train_split=train
done
