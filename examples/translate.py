"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import sys
sys.path.append('.')
sys.path.append('..')
import argparse
from pathlib import Path
import torch
from radiance_fields.qff import QFFRadianceField
from tqdm import tqdm
from tltorch import CPTensor, DenseTensor
from tensorly.decomposition import parafac, CP, parafac2
import tensorly as tl

def qff_2_to_qff_3(qff_2, qff_3, num_iters=50000):
    F = qff_2.num_freqs
    C = qff_2.num_feats
    Q = qff_2.num_quants
    R = qff_2.num_corrs
    vx, vy, vz = qff_2.qff_vector.view(F*2, 3, C, R, Q).chunk(3, 1)
    pyz, pxz, pxy = qff_2.qff_plane.view(F*2, 3, C, R, Q, Q).chunk(3, 1)

    f = vx.view(F*2, C, R, 1, 1, Q) * pyz.view(F*2, C, R, Q, Q, 1) + \
        vy.view(F*2, C, R, 1, Q, 1) * pxz.view(F*2, C, R, Q, 1, Q) + \
        vz.view(F*2, C, R, Q, 1, 1) * pxy.view(F*2, C, R, 1, Q, Q)
    qv = f.sum(2).view(F*2, C, Q, Q, Q)
    qvol = qff_3.qff_volume
    qvol.data.copy_(qv)

def qff_3_to_qff_1(qff_3, qff_1, num_iters=50000):
    # optimizer = torch.optim.Adam(qff_1.parameters(), lr=1e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    F = qff_1.num_freqs
    C = qff_1.num_feats
    R = qff_1.num_corrs
    Q = qff_1.num_quants
    qvol = qff_3.qff_volume.view(F*2, C, Q, Q, Q).detach()
    qvecs = []
    for f in tqdm(range(F*2), leave=False):
        qcvecs = []
        for c in tqdm(range(C), leave=False):
            qv = qvol[f, c]
            qfactors = parafac(qv, rank=R, verbose=True)
            qz, qy, qx = qfactors.factors
            cpt = tl.cp_to_tensor(qfactors)
            print((cpt - qv).abs().mean())
            qcvecs.append(torch.stack([qx, qy, qz], dim=0).permute(0, 2, 1))
        qvecs.append(torch.stack(qcvecs, dim=1))
    qff_1.qff_vector.copy_(torch.stack(qvecs, dim=0).view(F*2, 3, C*R, Q))


def qff_1_to_qff_3(qff_1, qff_3, num_iters=50000):
    F = qff_1.num_freqs
    C = qff_1.num_feats
    R = qff_1.num_corrs
    Q = qff_1.num_quants
    qvec = qff_1.qff_vector.view(F*2, 3, C, R, Q)
    qx = qvec[:,  0].view(F*2, C, R, 1, 1, Q)
    qy = qvec[:,  1].view(F*2, C, R, 1, Q, 1)
    qz = qvec[:,  2].view(F*2, C, R, Q, 1, 1)
    qv = (qx * qy * qz).view(F*2, C, R, Q, Q, Q).sum(2)
    qvol = qff_3.qff_volume
    qvol.data.copy_(qv)

def main(args):
    device = args.device
    max_steps = args.max_steps
    input_model_path = args.input_model_path
    ckpt = torch.load(input_model_path)
    model_states = ckpt['radiance_field_state_dict']
    freqs = model_states['encoder.freqs']
    num_freqs = freqs.shape[0]
    min_log2_freq = freqs[0].log2().item()
    max_log2_freq = freqs[-1].log2().item()
    num_out_features = model_states['geom_mlp.0.weight'].shape[1] 
    if 'encoder.qff_plane' in model_states:
        qff_type = 2
        qp = model_states['encoder.qff_plane']
        num_quants = qp.shape[2]
        num_channels = qp.shape[1]
        num_features = num_out_features // (num_freqs * 2)
        num_corrs = num_channels // num_features
    elif 'encoder.qff_volume' in model_states:
        qff_type = 3
        qv = model_states['encoder.qff_volume']
        num_corrs = 1
        num_quants = qv.shape[2]
        num_features = qv.shape[1]
    elif 'encoder.qff_vector' in model_states:
        qff_type = 1
        qv = model_states['encoder.qff_vector']
        num_quants = qv.shape[2]
        num_channels = qv.shape[1]
        num_features = num_out_features // (num_freqs * 2)
        num_corrs = num_channels // num_features
    else:
        raise NotImplementedError
    if args.output_model_type == qff_type:
        print(f"Input and output model types are the same: {qff_type}")
        return
    
    orig_model = QFFRadianceField(
        qff_type=qff_type, 
        num_quants=num_quants, 
        num_features=num_features,  
        min_log2_freq=min_log2_freq,
        max_log2_freq=max_log2_freq,
        num_freqs=num_freqs,
        num_corrs=num_corrs,
    )
    orig_model.load_state_dict(model_states)
    orig_model = orig_model.to(device)
    new_model = QFFRadianceField(
        qff_type=args.output_model_type, 
        num_quants=num_quants, 
        num_features=num_features,  
        min_log2_freq=min_log2_freq,
        max_log2_freq=max_log2_freq,
        num_freqs=num_freqs,
        num_corrs=args.num_corrs,
    )
    new_model = new_model.to(device)
    # translate QFF weights
    if qff_type == 3:
        # volume to vector / planes (compression)
        if(args.output_model_type == 1):
            qff_3_to_qff_1(orig_model.encoder, new_model.encoder, max_steps)
        elif(args.output_model_type == 2):
            qff_3_to_qff_2(orig_model.encoder, new_model.encoder, max_steps)
    else:
        # vector / planes to volume (extraction)
        if(qff_type == 1):
            qff_1_to_qff_3(orig_model.encoder, new_model.encoder, max_steps)
        elif(qff_type == 2):
            qff_2_to_qff_3(orig_model.encoder, new_model.encoder, max_steps)

    new_states = new_model.state_dict()
    for state in ['geom_mlp.0.weight', 'geom_mlp.2.weight', 'color_mlp.0.weight', 'color_mlp.2.weight', 'color_mlp.4.weight']:
        new_states[state] = model_states[state]

    ckpt['radiance_field_state_dict'] = new_states
    torch.save(ckpt, args.output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument( "--device", type=str, default="cuda", help="which device to use")
    parser.add_argument( "--max_steps", type=int, default=2500)
    parser.add_argument( "--num_corrs", type=int, default=8)
    parser.add_argument( "--input_model_path", type=str)
    parser.add_argument( "--output_model_path", type=str)
    parser.add_argument( "--output_model_type", type=int, default=3, choices=[1, 2, 3])

    args = parser.parse_args()
    main(args)