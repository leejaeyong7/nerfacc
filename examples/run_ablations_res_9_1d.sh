SCENE_IDS=(mic lego chair drums ficus hotdog materials ship)
RES=9
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_1d_l7_f${RES}_d2_$SCENE --scene=$SCENE --train_split=train --model_type=1d --log2_res=$RES
done

