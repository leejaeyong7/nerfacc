SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)
RESS=(4 10)
for SCENE in ${SCENE_IDS[@]}; do
    for RES in ${RESS[@]}; do
        python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_1d_l7_f${RES}_d2_$SCENE --scene=$SCENE --train_split=train --model_type=1d --log2_res=$RES
        python train_freqs.py --data_path=data/nerf_synthetic --run_name=freq_2d_l7_f${RES}_d2_$SCENE --scene=$SCENE --train_split=train --model_type=2d --log2_res=$RES
    done
done

