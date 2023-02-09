# SCENES=(courtyard  delivery_area  electro  facade  kicker  meadow  office  pipes  playground  relief  relief_2  terrace  terrains)
# for SCENE in ${SCENES[@]}; do
#     python train_freqs_eth3d.py --data_path=/mnt/data0/eth3d_processed --log_path=logs --run_name=freq_2d_$SCENE --scene=$SCENE --model_type=2d
#
# done
#
#
SCENE_IDS=(chair drums ficus hotdog lego materials mic ship)

for SCENE in ${SCENE_IDS[@]}; do
    python train_qff_ngp.py \
      --data_path=data/nerf_synthetic \
      --log_path=logs \
      --run_name=qff-$SCENE --scene=$SCENE \
      --train_split=train \
      --run_name=qff-lite-0-3-4-4-80-15-$SCENE \
      -f 4 -n 0 -x 4 -c 4 -q 80 -d 15
done

