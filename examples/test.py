runs_raw = '''freq_1d_l5_f16_d8_lego_model_step_50000.pth
freq_1d_l7_f9_d2_lego_model_step_50000.pth
freq_1d_lego_freq_2d_model_step_50000.pth
freq_2d_l7_f16_d2_lego_model_step_50000.pth
freq_2d_l7_f5_d2_lego_model_step_50000.pth
freq_2d_l7_f9_d2_lego_model_step_50000.pth
freq_2d_lego_freq_2d_model_step_50000.pth'''

import torch
runs = runs_raw.split('\n')
print(runs)

for run in runs:
    ckpt = torch.load(f'checkpoints/{run}', 'cpu')
    print(run, sum([v.numel() for v in ckpt.values()]))
