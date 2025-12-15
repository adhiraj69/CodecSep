"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec
"""
import sys
import os


from src.models.codecsep import CodecSep

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



import sys
import json
import argparse
import importlib
import math
import julius
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from collections import namedtuple

from src import utils

import torch
import torchaudio
from accelerate import Accelerator







accelerator = Accelerator()

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ret-dir', type=str, default='model-checkpoints/CodecSep_AudioCaps_USS_Weights', help='Training result directory')
parser.add_argument('--csv-path', type=str, default='manifest/test_dnr.csv', help='csv file to test')
parser.add_argument('--data-sr', type=int, default=[44100], nargs='+', help='list of sampling rate in test files')
parser.add_argument('--length', type=int, default=10, help='audio length')
parser.add_argument('--visqol-mode', type=str, default='speech', choices=['speech', 'audio'], help='visqol mode')
parser.add_argument('--threshold', type=float, default=0.4, help='threshold of silence part to drop audio')
parser.add_argument('--fast', action='store_true', help='fast eval, disable visqol computation')

# parse
args = parser.parse_args()
ret_dir = Path(args.ret_dir)
csv_path = Path(args.csv_path)
length = args.length
#visqol_mode = args.visqol_mode
threshold = args.threshold
#use_visqol = not args.fast
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# read config
cfg_filepath = ret_dir / '.hydra' / 'config.yaml'
cfg = OmegaConf.load(cfg_filepath)



sample_rate = cfg.sampling_rate
chunk_len = sample_rate * length

# init julius resample
resample_pool = dict()
for sr in args.data_sr:
    old_sr = 24000
    new_sr = sample_rate
    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    resample_pool[24000] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

resample_pool2 = dict()
for sr in args.data_sr:
    old_sr = 24000
    new_sr = 16000
    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    resample_pool2[24000] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)


resample_pool3 = dict()
for sr in args.data_sr:
    old_sr = 16000
    new_sr = sample_rate
    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    resample_pool3[16000] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)


#import lib

model_name = cfg.model.codecsep_params.pop('name')
module_path = str(ret_dir / 'backup_src' / 'models').replace('/', '.')
try:
    load_model = importlib.import_module(module_path)
    if model_name != 'CodecSep':
        net_class = getattr(load_model, f'{model_name}')
        print('Load model from ckpt')
    else:
        print('MODEL', model_name)
        net_class = CodecSep #(load_model, f'{model_name}')
        print('Load model from ckpt')
except:
    from src import models
    if model_name != 'CodecSep':
        net_class = getattr(models, f'{model_name}')
        print('Load model from source code')
    else:
        print('MODEL', model_name)
        net_class = CodecSep
        print('Load model from source code')

#      device=device)
model_cfg = cfg.model.codecsep_params
model = net_class(sample_rate=sample_rate, **model_cfg)





total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Total params: {total_params:.2f} Mb')
print('Model sampling rate: {} Hz'.format(sample_rate))

ckpt_finalpath = ret_dir / 'ckpt_final' / 'ckpt_model_final.pth'
state_dict = torch.load(ckpt_finalpath, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
#print(f'ckpt path: {ckpt_finalpath}')
print(f'Model weights load successfully...')

# prepare metrics
loss_cfg = cfg.training.loss


# prepare data transform
transform_cfg = cfg.training.transform
volume_norm = utils.VolumeNorm(sample_rate=sample_rate)
def _data_transform(batch, transform_cfg, valid_tracks=['speech'], norm_var=0):
    peak_norm = utils.db_to_gain(transform_cfg.peak_norm_db)
    mix_max_peak = torch.zeros_like(batch['speech'])[...,:1] # (bs, C, 1)

    # volume norm for each track
    for track in valid_tracks:
        batch[track] = volume_norm(signal=batch[track],
                                    target_loudness=transform_cfg.lufs_norm_db[track],
                                    var=norm_var)
        # peak value
        peak = batch[track].abs().max(dim=-1, keepdims=True)[0]
        mix_max_peak = torch.maximum(peak, mix_max_peak)
    
    # peak norm
    peak_gain = torch.ones_like(mix_max_peak) # (bs, C, 1)
    peak_gain[mix_max_peak > peak_norm] = peak_norm / mix_max_peak[mix_max_peak > peak_norm]
    
    # build mix
    batch['mix'] = torch.zeros_like(batch['speech'])
    for track in valid_tracks:
        batch[track] *= peak_gain
        batch['mix'] += batch[track]

    # mix volum norm
    batch['mix'], mix_gain = volume_norm(signal=batch['mix'],
                                        target_loudness=transform_cfg.lufs_norm_db['mix'],
                                        var=norm_var,
                                        return_gain=True)
    
    # norm each track
    for track in valid_tracks:
        batch[track] *= mix_gain[:, None, None]

    batch['valid_tracks'] = valid_tracks
    batch['random_swap'] = False

    return batch


# define mask separation
sep_norm = utils.WavSepMagNorm()

# define STFT params
STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length", "window_type", "padding_type"],
)
stft_params = STFTParams(
                window_length=1024,
                hop_length=256,
                window_type="hann",
                padding_type="reflect",
            )

# run eval
tracks = ['speech', 'music', 'sfx']
print('Model tracks: {}'.format(tracks))
test_tracks = ['mix'] + [f'{t}_rec' for t in tracks] + [f'{t}_sep_mask' for t in tracks]
test_results = {t: {} for t in test_tracks}
metadata = pd.read_csv(csv_path)
import math 

cnt = 0 
dump_dir = 'CodecSep_AudioCaps_USS_inference_AudioCaps/'

import os

os.makedirs(dump_dir, exist_ok = True)


import torch
from torch.utils.data import Dataset,DataLoader, RandomSampler


import os



class PTFileDataset(Dataset):
    def __init__(self, directory):
        """
        Custom dataset for loading batches stored in .pt files.

        Args:
            directory (str): Path to the directory containing .pt files.
        """
        self.directory = directory
        self.files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .pt file and return its contents.

        Args:
            idx (int): Index of the file to load.
        
        Returns:
            dict or tensor: The loaded data.
        """
        file_path = self.files[idx]
        data = torch.load(file_path)  # Load the .pt file
        return data  # This assumes the .pt file stores a dictionary or tensor
                                                                                                

# Path to your directory containing .pt files
data_dir = "./datasets/audiocaps_test"



# Create dataset instance
dataset = PTFileDataset(data_dir)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
track2id = {tracks[i]: i  for i in range(len(tracks)) }

cnt = 0
for batch in tqdm(dataloader):

    audio_id =cnt 
    #if audio_id > 32:
    #    break
    starts = batch['start_time'] 

    annots = batch['caption'] 


    audios = batch['audio']['array'].squeeze(0)



    sr = 24000


    batch = {}
    # read data
    for t in tracks:
        x, sr = audios[track2id[t]], sr

        x = x.mean(dim=0)[...,:] # start: end]
        #print(x.shape)

        if sr != 16000:

            resample_pool2[sr] = resample_pool2[sr].to(torch.float32)
            x = x.to(torch.float32)

            x = resample_pool2[sr](x)
        
  

        
        batch[t] = x
        
        audio_len = x.shape[-1]

    # clip audio
    #for j, k in enumerate(range(0, audio_len-chunk_len+1, chunk_len)):
    clip_id = f'{audio_id}'
    eval_batch = {}
    mask = {}
    for t in tracks:
            audio_clip = batch[t]#[k:k+chunk_len]

            # silent audio detection
            audio_energy = torch.stft(audio_clip, n_fft=stft_params.window_length, hop_length=stft_params.hop_length, 
                             win_length=stft_params.window_length,
                             window=torch.hann_window(stft_params.window_length, device='cpu'),
                             pad_mode=stft_params.padding_type, center=True, onesided=True, return_complex=True).abs().sum(dim=0)


            count = sum(1 for item in audio_energy if item > 1e-6)
            silence_detect = count < threshold * len(audio_energy)
            

            mask[f'{t}_rec'] = silence_detect
            mask[f'{t}_sep_mask'] = silence_detect
      
            



            eval_batch[t] = audio_clip.reshape(1,1,-1).to(device)




            eval_batch[f'{t}/prompt'] =  annots[track2id[t]][0].lower()



    mask['mix'] = all(mask.values())

        # data transform
        # eval_batch = _data_transform(eval_batch, transform_cfg=transform_cfg, valid_tracks=tracks, norm_var=0)
    eval_batch['mix'] = eval_batch['speech']+eval_batch['music']+eval_batch['sfx']
    eval_batch['valid_tracks'] = tracks
    eval_batch['random_swap'] = False
        
        # mixture forward
    with torch.no_grad():
            output_audio = model.evaluate((eval_batch['mix'], [eval_batch[f'{t}/prompt'] for t in tracks]),
                                          output_tracks=['mix']+tracks)

             
        # Eval mix reconstruction
    est = output_audio[:, 0].unsqueeze(1)

    torch.save(est.cpu(),f'{dump_dir}mix-est-{cnt}.pt' )
        #Data_dict['mix/est'].append(est.cpu().numpy()) 
    ref = eval_batch['mix']
        #print('ref', ref.shape, est.shape)
    torch.save(ref.cpu(), f'{dump_dir}mix-ref-{cnt}.pt')

        # Eval separation using synthesizer (decoder)
    for p, t in enumerate(tracks):


            ref = eval_batch[t]

            torch.save(ref.cpu(),f'{dump_dir}{t}-ref-{cnt}.pt' )

        # Eval separation using mask
    mix = eval_batch['mix'].unsqueeze(2)

        #print(mix.shape)
    signal_sep = output_audio[:,1:].unsqueeze(2)
        #print(signal_sep.shape)
        
    all_sep_mask_norm = sep_norm(mix, signal_sep)
        #print(all_sep_mask_norm)
    for p, t in enumerate(tracks):
    
            
            est = all_sep_mask_norm[:,p]
            ref = eval_batch[t]
           # print('track est', est.shape)
            torch.save(est.cpu(), f'{dump_dir}{t}_sep_mask-est-{cnt}.pt')

            ref = ref[...,:est.shape[-1]] # stft + istft. shorter

        # Evaluate reconstruction of single track
    for p, t in enumerate(tracks):
            # single track forward
            with torch.no_grad():
                output_audio = model.evaluate((eval_batch[f'{t}'], [eval_batch[f'{t_}/prompt'] for t_ in tracks]),
                                                            output_tracks=[t])

            est = output_audio
            torch.save(est.cpu(), f'{dump_dir}{t}_rec-est-{cnt}.pt')

            ref = eval_batch[t]


    cnt +=1
