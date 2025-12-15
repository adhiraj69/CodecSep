
"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec
"""


import math
import numpy as np
import pandas as pd
import julius
from pathlib import Path
from omegaconf import DictConfig
from typing import List
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from accelerate.logging import get_logger
import random
from datasets import load_dataset
import itertools
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch
from torch.utils.data import Dataset,DataLoader, RandomSampler

#sampler = RandomSampler(ds['test'], num_samples=3, replacement=False)
#dataloader = DataLoader(ds['test'], batch_size=3, sampler=sampler)
import os



class PTFileDataset(Dataset):
    def __init__(self, directory, dev = False):
        """
        Custom dataset for loading batches stored in .pt files.

        Args:
            directory (str): Path to the directory containing .pt files.
        """
        self.directory = directory
        self.files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')])

        if dev: 
            self.files = self.files[:32]

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




def collate_fn(batch):
    max_len = max(len(item["audio"]["array"]) for item in batch)  # Find max audio length

    for item in batch:
        audio_tensor = torch.tensor(item["audio"]["array"])
        if len(audio_tensor) < max_len:
            # Pad with zeros
            pad_size = max_len - len(audio_tensor)
            audio_tensor = torch.cat([audio_tensor, torch.zeros(pad_size)], dim=0)
        else:
            # Truncate
            audio_tensor = audio_tensor[:max_len]

        # Replace the audio array with the padded tensor
        item["audio"]["array"] = audio_tensor

    # Return structured batch with all keys intact
    return {
        key: ([item[key] for item in batch] if key != "audio" else 
              { 
                  "path": [item["audio"]["path"] for item in batch],
                  "array": torch.stack([item["audio"]["array"] for item in batch]).unsqueeze(1),
                  "sampling_rate": [item["audio"]["sampling_rate"] for item in batch]  # Keep as list
              })  
        for key in batch[0]
    }        
      

logger = get_logger(__name__)



class AudioCapsDatasetAudioTrain(Dataset):
    def __init__(self,
        sample_rate: int,
        #speech: List[str],
        #music: List[str],
        #sfx: List[str],
        n_examples: int = 10000000,
        chunk_size: float = 2.0,
        trim_silence: bool = False,
        use_background: bool = False,
        num_background: int = 3,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        self.length = n_examples # pseudo dataset length
        self.chunk_size = chunk_size # negative for entire sentence
        self.trim_silence = trim_silence

        self.use_background= use_background
        self.num_background = num_background 

        ds = load_dataset("OpenSound/AudioCaps", split = 'train')
        ds = ds.shuffle(seed=42)
        self.dataset = ds
        #sampler = RandomSampler(ds, replacement=False)
        #self.dataloader = DataLoader(ds, batch_size=3, sampler = sampler, collate_fn= collate_fn)

        # manifest

        self.tracks = ['music', 'speech', 'sfx']

        self.track2id = {f'{self.tracks[i]}':i for i in range(len(self.tracks)) }
        
        # check valid samples
        self.resample_pool = dict()
        self.metadata_dict = dict()
        self.lens_dict = dict()
        
        sr = 24000
        if sr not in self.resample_pool.keys():
                                old_sr = sr
                                new_sr = self.sample_rate
                                gcd = math.gcd(old_sr, new_sr)
                                old_sr = old_sr // gcd
                                new_sr = new_sr // gcd
                                self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)
        self.resample_pool[sr] = self.resample_pool[sr].to(torch.float32)




        logger.info('Resample pool: {}'.format(list(self.resample_pool.keys())))


    def __len__(self):
        return self.length # can be any number




    def sample_phrase_or_full(self, annotation):
        # Split by comma and strip whitespace
        phrases = [p.strip() for p in annotation.split(",") if p.strip()]

        if len(phrases) == 1:
            return phrases[0]  # only one phrase, return it

        # 50% chance: return one random phrase
        # 50% chance: return the whole annotation
        if random.random() < 0.5:
            return random.choice(phrases)
        else:
            return ", ".join(phrases)

    def __getitem__(self, idx:int):



        batch = {}


        for t in self.tracks:
            idx = np.random.randint(len(self.dataset))
            wav_info =self.dataset[idx]#next(itertools.islice(self.dataloader, idx, None))
            #print(wav_info)
            annots= wav_info['caption']

            sources = wav_info['audio']['array']
            #sources = np.expand_dims(sources,0)
            #sources = np.expand_dims(sources,0)
            #print(sources.shape)
            sampling_rates = wav_info['audio']['sampling_rate']

            chunk_len = int(self.chunk_size * sampling_rates)


            audio_lengths = len(sources) #wav_info['audio_length']
            #print('audiolength', audio_lengths)

            if int(audio_lengths) > chunk_len:
                start = np.random.randint(0, int(audio_lengths) - chunk_len + 1)
            else:

                start = 0

            sources = torch.tensor(sources)
            sources = F.pad(sources, (0, max(0, chunk_len - sources.shape[-1])))
        #for t in self.tracks:    
            batch[f'{t}/prompt'] = annots.lower()
            
            #print('start shape',sources.shape)

            #print(start,start+ chunk_len)
            sources = sources[start:start + chunk_len]

            #print('end shape',sources.shape)

            # load file
            # single channel

            x = sources.unsqueeze(0)#.unsqueeze(0)#[self.track2id[t]]
            #print('Source', x.shape)
            x = x.mean(dim=0, keepdim=True)
            sr = sampling_rates#[self.track2id[t]]
            # resample

            #resample_pool[sr] = resample_pool[sr].to(torch.float32)
            x = x.to(torch.float32)


            if sr != self.sample_rate:
                x = self.resample_pool[sr](x)

            batch[t] = x

        batch['background'] = torch.zeros_like(batch['speech'])

        if self.use_background:
            back_sources = []
            for i in range(self.num_background):


                batch_idx = np.random.randint(len(self.dataset))


                annots= wav_info['caption']

                sources = wav_info['audio']['array']
                #print(sources.shape)
                sampling_rates = wav_info['audio']['sampling_rate']
                chunk_len = int(self.chunk_size * sampling_rates)


                audio_lengths = len(sources) #wav_info['audio_length']
                #print('audiolength', audio_lengths)

                if int(audio_lengths) > chunk_len:
                    start = np.random.randint(0, int(audio_lengths) - chunk_len + 1)
                else:

                    start = 0

                sources = torch.tensor(sources)
                sources = F.pad(sources, (0, max(0, chunk_len - sources.shape[-1])))
                sources = sources[start:start + chunk_len]
                back_sources.append(sources)
            background = sum(back_sources)

            batch['background'] = background


        return batch



class AudioCapsDatasetAudioVal(Dataset):
    def __init__(self,
        sample_rate: int,
        tsv_filepath: str,
        chunk_size: float = 5.0,
        dev: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        #self.tsv_filepath = Path(tsv_filepath)
        self.chunk_size = chunk_size # negative for entire sentence
        self.resample_pool = dict()
        
        val_dir = tsv_filepath   # './datasets/AudioCaps_Valid'
        # read manifest tsv file

        self.dataset = PTFileDataset(val_dir, dev)

        tracks = ['speech', 'music', 'sfx']

        self.track2id = {tracks[i]: i  for i in range(len(tracks)) }

        # audio lengths
        sr = 24000 #int(metadata.at[row_idx, 'sr'])
        if sr not in self.resample_pool.keys():
                    old_sr = sr
                    new_sr = self.sample_rate
                    gcd = math.gcd(old_sr, new_sr)
                    old_sr = old_sr // gcd
                    new_sr = new_sr // gcd
                    self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)


    def __len__(self):
        return len(self.dataset)




    def sample_phrase_or_full(self, annotation):
        # Split by comma and strip whitespace
        phrases = [p.strip() for p in annotation.split(",") if p.strip()]

        if len(phrases) == 1:
            return phrases[0]  # only one phrase, return it

        # 50% chance: return one random phrase
        # 50% chance: return the whole annotation
        if random.random() < 0.5:
            return random.choice(phrases)
        else:
            return ", ".join(phrases)

    def __getitem__(self, idx:int):

        wav_info = self.dataset[idx]
        
        annots= wav_info['caption']

        sources = wav_info['audio']['array']
            #sources = np.expand_dims(sources,0)
            #sources = np.expand_dims(sources,0)
            #print(sources.shape)
        sampling_rates = wav_info['audio']['sampling_rate']

        chunk_len = int(self.chunk_size * sampling_rates[0])


        audio_lengths = len(sources) #wav_info['audio_length']
            #print('audiolength', audio_lengths)
        #sentence = annots["annotation"].str.cat(sep=",")
 
        batch = {} 
        #'prompt' : sentence.lower()}
        for track in ['mix', 'speech', 'music', 'sfx']:
            if track != 'mix':
                #print(sources.shape)
                x= sources[self.track2id[track]]

                #print(x.shape)


                x = x[:,:chunk_len]#.unsqueeze(0)


                x = F.pad(x, (0, max(0, chunk_len - x.shape[-1])))

                #print(x.shape)
                sr = sampling_rates[self.track2id[track]]


                x = x.mean(dim=0, keepdim=True)
                #print(x.shape)
                # resample

                x = x.to(torch.float32)
                if sr != self.sample_rate:
                    x = self.resample_pool[sr](x)
                batch[track] = x
                batch[f'{track}/prompt'] = annots[self.track2id[track]].lower()
            else:
                x = sources.sum(0)


                x = x[:,:chunk_len]#.unsqueeze(0)


                x = F.pad(x, (0, max(0, chunk_len - x.shape[-1])))

                #print(x.shape)
                sr = sampling_rates[self.track2id['sfx']]


                x = x.mean(dim=0, keepdim=True)
                #print(x.shape)
                # resample

                x = x.to(torch.float32)
                if sr != self.sample_rate:
                    x = self.resample_pool[sr](x)
                batch[track] = x#.squeeze()

        batch['background'] = torch.zeros_like(batch['speech']).unsqueeze(0)            
        #print('yo', batch['background'].shape)

        return batch
    
