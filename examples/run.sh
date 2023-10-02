python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 4 --num_features 4 --num_quants 128 --qff_type 3 --net_depth 2
<<com
# 31.95
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 6 --num_features 0 --num_quants 0 --qff_type 0 --net_depth 8 
# 28.52
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 4 --num_features 32 --num_quants 128 --qff_type 1 --net_depth 2 
# 28.75
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 8 --num_features 16 --num_quants 128 --qff_type 1 --net_depth 2 
# 28.01
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 4 --num_features 32 --num_quants 80  --qff_type 1 --net_depth 2 
# 28.36
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 4 --num_features 32 --num_quants 256 --qff_type 1 --net_depth 2 
# 28.96
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 4 --num_features 32 --num_quants 128 --qff_type 1 --net_depth 3 
# 33.98
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 8 --num_features 8 --num_quants 128 --qff_type 2 --net_depth 2
# 
python  train_qff_nerf.py --data_path=data/nerf_synthetic --scene=lego --train_split=train --log_path=logs --num_freqs 8 --num_features 8 --num_quants 128 --qff_type 2 --net_depth 2
com

