SCENES=(courtyard  delivery_area  electro  facade  kicker  meadow  office  pipes  playground  relief  relief_2  terrace  terrains)
for SCENE in ${SCENES[@]}; do
    python train_freqs_eth3d.py --data_path=/mnt/data0/eth3d_processed --log_path=logs --run_name=freq_2d_$SCENE --scene=$SCENE --model_type=2d
done


# for SCENE in ${SCENE_IDS[@]}; do
#     python train_freqs.py --data_path=data/nerf_synthetic --log_path=logs --run_name=freq_1d_$SCENE --scene=$SCENE --train_split=train --model_type=1d
# done

