SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
for SCENE in ${SCENE_IDS[@]}; do
    python eval_mlp_nerf.py --data_path=/home/jae/dev/data/nerf_synthetic --scene $SCENE --ckpt_file checkpoints-old/${SCENE}_van_model.pth --grid_file checkpoints-old/${SCENE}_van_grid.pth --output_folder=outputs/${SCENE}_van
done

