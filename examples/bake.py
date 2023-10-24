import numpy as np
import cbor2
import math

def prepare_bake_data(model):
    F = model.encoder.num_freqs
    C = model.encoder.num_feats
    Q = model.encoder.num_quants
    R = model.encoder.num_corrs
    qff_type = model.qff_type

    # uses opencv convention by default.
    up = [0, 0, 1]
    c = math.cos(math.pi / 4)
    pose = [
        [1, 0, 0, 0], 
        [0, c, c, 0], 
        [0, -c, c, math.pi / 4], 
        [0, 0, 0, 1]
    ]

    freqs = model.encoder.freqs.detach().cpu().numpy()
    weights = model.mlp.weight.data.T.view(F*2, C, 8).permute(0, 2, 1).cpu().numpy()

    # F2 x rgba x C=4
    qff_rgba_buffer = weights[:, :4].tobytes()

    # F2 x C x dxyz
    qff_dxyz_buffer = weights[:, 4:].transpose(0, 2, 1).tobytes()

    qff_buffer, qff_mean = model.get_qff_buffer()


    data_to_write = {
        'freqs': freqs.tolist(),
        'n_freqs': len(freqs),
        'n_feats': C,
        'n_quants': Q,
        'rank': R,
        'qff_mean': qff_mean,
        'render_step': 5e-3,
        'up': up,
        'initial_pose': pose,
        'qff_type': qff_type,
        'qff_buffer': qff_buffer,
        'qff_rgba_buffer': qff_rgba_buffer,
        'qff_dxyz_buffer': qff_dxyz_buffer, 
    }

    return data_to_write

    
def bake(model, bake_path):
    data_to_write = prepare_bake_data(model)

    with open(bake_path, 'wb') as f:
        f.write(cbor2.dumps(data_to_write))
