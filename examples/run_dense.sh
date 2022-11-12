SCENE_IDS=(lego)
for SCENE in ${SCENE_IDS[@]}; do
    python train_freqs_dense.py --data_path=data/nerf_synthetic --run_name=freq_1d_300_l7_f16_d8_$SCENE --scene=$SCENE --train_split=train --model_type=1d
done
