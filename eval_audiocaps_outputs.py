"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec
"""


import sys
import os





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
from src.metrics import (
    VisqolMetric,
    SingleSrcNegSDR,
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
)

import torch
import torchaudio
from accelerate import Accelerator
from audiotools import AudioSignal
import math


accelerator = Accelerator()

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ret-dir', type=str, default='model-checkpoints/CodecSep_AudioCaps_USS_Weights', help='Training result directory')
parser.add_argument('--csv-path', type=str, default='./manifest/test_dnr.csv', help='csv file to test')
parser.add_argument('--data-sr', type=int, default=[24000], nargs='+', help='list of sampling rate in test files')
parser.add_argument('--length', type=int, default=10, help='audio length')
parser.add_argument('--visqol-mode', type=str, default='speech', choices=['speech', 'audio'], help='visqol mode')
parser.add_argument('--threshold', type=float, default=0.4, help='threshold of silence part to drop audio')
parser.add_argument('--fast', action='store_true', help='fast eval, disable visqol computation')

# parse
args = parser.parse_args()
ret_dir = Path(args.ret_dir)
csv_path = Path(args.csv_path)
length = args.length
visqol_mode = args.visqol_mode
threshold = args.threshold
use_visqol = not args.fast
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# read config
cfg_filepath = ret_dir / '.hydra' / 'config.yaml'
cfg = OmegaConf.load(cfg_filepath)
sample_rate = cfg.sampling_rate
chunk_len = sample_rate * length

# init julius resample
resample_pool = dict()
for sr in args.data_sr:
    old_sr = sr
    new_sr = sample_rate
    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)



# prepare metrics
loss_cfg = cfg.training.loss
metric_stft = MultiScaleSTFTLoss(**loss_cfg.MultiScaleSTFTLoss)
metric_mel = MelSpectrogramLoss(**loss_cfg.MelSpectrogramLoss)
metric_sisdr = SingleSrcNegSDR(sdr_type='sisdr')
metric_visqol = VisqolMetric(mode=visqol_mode)

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
#model.tracks
print('Model tracks: {}'.format(tracks))
test_tracks = ['mix'] + [f'{t}_rec' for t in tracks] +  [f'{t}_sep_mask' for t in tracks]
test_results = {t: {} for t in test_tracks}
metadata = pd.read_csv(csv_path)
cnt = 0
dump_dir = 'CodecSep_AudioCaps_USS_inference_AudioCaps/'

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



data_dir = "./datasets/audiocaps_test"



# Create dataset instance
dataset = PTFileDataset(data_dir)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
track2id = {tracks[i]: i  for i in range(len(tracks)) }

cnt = 0
for batch in tqdm(dataloader):
    audio_id =cnt #wav_info['id']

    #if cnt > 1:

    #    break
    sr = 24000
    #sr = float(sr) 
    starts = batch['start_time']

    annots = batch['caption']


    audios = batch['audio']['array'].squeeze(0)

    batch = {}
    # read data
    for t in tracks:
        x, sr = audios[track2id[t]], sr  #torchaudio.load(wav_info[t])


        #print(x.shape) 
        x = x.mean(dim=0)[..., :]#[..., start: end]
   
        #print(x.shape)
        if sr != sample_rate:
            resample_pool[sr] = resample_pool[sr].to(torch.float32)
            x = x.to(torch.float32)



            x = resample_pool[sr](x)

        batch[t] = x
        audio_len = x.shape[-1]

    # clip audio
   # for j, k in enumerate(range(0, audio_len-chunk_len+1, chunk_len)):
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
        
    mask['mix'] = all(mask.values())

        # data transform
        # eval_batch = _data_transform(eval_batch, transform_cfg=transform_cfg, valid_tracks=tracks, norm_var=0)
    eval_batch['mix'] = eval_batch['speech']+eval_batch['music']+eval_batch['sfx']
    eval_batch['valid_tracks'] = tracks
    eval_batch['random_swap'] = False
        

        # Eval mix reconstruction
    est = torch.load(f'{dump_dir}mix-est-{cnt}.pt').to(device)   #output_audio[:, 0].unsqueeze(1)
    ref = torch.load(f'{dump_dir}mix-ref-{cnt}.pt').to(device) #eval_batch['mix']
    test_results['mix'][clip_id] = {}
    if mask['mix']:
            test_results['mix'][clip_id]['stft'] = None
            test_results['mix'][clip_id]['mel'] = None
            test_results['mix'][clip_id]['sisdr'] = None
            if use_visqol:
                test_results['mix'][clip_id]['visqol'] = None
    else:
            test_results['mix'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
            test_results['mix'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
            test_results['mix'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
            if use_visqol:
                test_results['mix'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)

 
        
        # Eval separation using mask

    for p, t in enumerate(tracks):
            est = torch.load(f'{dump_dir}{t}_sep_mask-est-{cnt}.pt') #all_sep_mask_norm[:,p]
            ref = torch.load(f'{dump_dir}{t}-ref-{cnt}.pt') #eval_batch[t]
            ref = ref[...,:est.shape[-1]] # stft + istft. shorter
            # breakpoint()
            test_results[f'{t}_sep_mask'][clip_id] = {}
            if mask[f'{t}_sep_mask']:
                test_results[f'{t}_sep_mask'][clip_id]['stft'] = None
                test_results[f'{t}_sep_mask'][clip_id]['mel'] = None
                test_results[f'{t}_sep_mask'][clip_id]['sisdr'] = None
                if use_visqol:
                    test_results[f'{t}_sep_mask'][clip_id]['visqol'] = None
                    print(test_results[f'{t}_sep_mask'][clip_id]['visqol'])

            else:
                test_results[f'{t}_sep_mask'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
                test_results[f'{t}_sep_mask'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
                test_results[f'{t}_sep_mask'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
                if use_visqol:
                    test_results[f'{t}_sep_mask'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)
                    #print(test_results[f'{t}_sep_mask'][clip_id]['visqol'])

        # Evaluate reconstruction of single track
    for p, t in enumerate(tracks):
            # single track forward


            est = torch.load(f'{dump_dir}{t}_rec-est-{cnt}.pt') #output_audio
            ref = torch.load(f'{dump_dir}{t}-ref-{cnt}.pt') #eval_batch[t]
#eval_batch[t]
            test_results[f'{t}_rec'][clip_id] = {}
            if mask[f'{t}_rec']:
                test_results[f'{t}_rec'][clip_id]['stft'] = None
                test_results[f'{t}_rec'][clip_id]['mel'] = None
                test_results[f'{t}_rec'][clip_id]['sisdr'] = None
                if use_visqol:
                    test_results[f'{t}_rec'][clip_id]['visqol'] = None
            else:
                test_results[f'{t}_rec'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
                test_results[f'{t}_rec'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
                test_results[f'{t}_rec'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()

                if use_visqol:
                    
                    test_results[f'{t}_rec'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)
                    #print(test_results[f'{t}_rec'][clip_id]['visqol'])
    cnt +=1





test_results['summary'] = {}

metrics_ = [ 'sisdr', 'visqol']

rec = [f'{t}_rec' for t in tracks] 

sep_mask = [f'{t}_sep_mask' for t in tracks]


keyz = {'rec':rec,  'sep_mask': sep_mask}

dict_full = {f'{t}': {f'{item}' : [] for item in metrics_} for t in ['rec',  'sep_mask']}

for track in test_tracks:
    test_results['summary'][track] = {}
    list_stft = []
    list_mel = []
    list_sisdr = []
    if use_visqol:
        list_visqol = []

    for metrics in test_results[track].values():
        list_stft.append(metrics['stft'])
        list_mel.append(metrics['mel'])
        list_sisdr.append(metrics['sisdr'])
        if use_visqol:
            list_visqol.append(metrics['visqol'])

    np_stft = np.array([x for x in list_stft if x is not None])
    np_mel = np.array([x for x in list_mel if x is not None])
    np_sisdr = np.array([x for x in list_sisdr if x is not None])
    if use_visqol:
        np_visqol = np.array([x for x in list_visqol if x is not None])

    stft_m, stft_std = np.mean(np_stft), np.std(np_stft)
    mel_m, mel_std = np.mean(np_mel), np.std(np_mel)
    sisdr_m, sisdr_std = np.mean(np_sisdr), np.std(np_sisdr)
    if use_visqol:
        visqol_m, visqol_std = np.mean(np_visqol), np.std(np_visqol)

   
    for t in keyz.keys():
    #if track in rec:
        if track in keyz[t]:


            dict_full[t]['sisdr'].extend( np_sisdr.tolist())


            if use_visqol:

                dict_full[t]['visqol'].extend(np_visqol.tolist())










for k in keyz.keys():



    sisdr_ = dict_full[k]['sisdr']

    if use_visqol:

        visqol_ = dict_full[k]['visqol']

    sisdr_m, sisdr_std = np.mean(sisdr_), np.std(sisdr_)
    if use_visqol:
            visqol_m, visqol_std = np.mean(visqol_), np.std(visqol_)
    

    print('='*80)
    print(f'{k}')
    print('Valid datapoint: {}/{}'.format(len(np_stft), len(list_stft)))

    print('SI-SDR: {:.2f} +/- {:.2f}'.format(sisdr_m, sisdr_std))
    if use_visqol:
            print('VisQOL: {:.2f} +/- {:.2f}'.format(visqol_m, visqol_std))





# save to json
json_filename = ret_dir / '{}_{}s_AudioCaps_USS.json'.format(csv_path.stem, length)
with open(json_filename, 'w') as f:
    json.dump(test_results, f, indent=1)
