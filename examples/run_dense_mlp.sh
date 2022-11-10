SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python train_mlp_nerf.py --data_path=data/nerf_synthetic --run_name=van_$SCENE --scene=$SCENE --train_split=train
done
