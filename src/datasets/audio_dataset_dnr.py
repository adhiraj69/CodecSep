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

logger = get_logger(__name__)



class DatasetAudioTrain(Dataset):
    def __init__(self,
        sample_rate: int,
        speech: List[str],
        music: List[str],
        sfx: List[str],
        n_examples: int = 10000000,
        chunk_size: float = 2.0,
        trim_silence: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        self.length = n_examples # pseudo dataset length
        self.chunk_size = chunk_size # negative for entire sentence
        self.trim_silence = trim_silence
        
        # manifest
        self.csv_files = {}
        self.csv_files['speech'] = [Path(filepath) for filepath in speech]
        self.csv_files['music'] = [Path(filepath) for filepath in music]
        self.csv_files['sfx'] = [Path(filepath) for filepath in sfx]

        # check valid samples
        self.resample_pool = dict()
        self.metadata_dict = dict()
        self.lens_dict = dict()
        for track, files in self.csv_files.items():
            logger.info(f"Track: {track}")
            orig_utt, orig_len, drop_utt, drop_len = 0, 0, 0, 0
            metadata_list = []
            for tsv_filepath in files:
                if not tsv_filepath.is_file():
                    logger.error('No tsv file found in: {}'.format(tsv_filepath))
                    continue
                else:
                    logger.info(f'Manifest filepath: {tsv_filepath}')
                    metadata = pd.read_csv(tsv_filepath)
                    if self.trim_silence:
                        wav_lens = (metadata['end'] - metadata['start']) / metadata['sr']
                    else:
                        wav_lens = metadata['length'] / metadata['sr']
                    # remove wav files that too short
                    orig_utt += len(metadata)
                    drop_rows = []
                    for row_idx in range(len(wav_lens)):
                        orig_len += wav_lens[row_idx]
                        if wav_lens[row_idx] < self.chunk_size:
                            drop_rows.append(row_idx)
                            drop_utt += 1
                            drop_len += wav_lens[row_idx]
                        else:
                            # prepare julius resample
                            sr = int(metadata.at[row_idx, 'sr'])
                            if sr not in self.resample_pool.keys():
                                old_sr = sr
                                new_sr = self.sample_rate
                                gcd = math.gcd(old_sr, new_sr)
                                old_sr = old_sr // gcd
                                new_sr = new_sr // gcd
                                self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

                    metadata = metadata.drop(drop_rows)
                    metadata_list.append(metadata)

            self.metadata_dict[track] = pd.concat(metadata_list, axis=0)
            self.lens_dict[track] = len(self.metadata_dict[track])
            
            logger.info("Drop {}/{} utterances ({:.2f}/{:.2f}h), shorter than {:.2f}s".format(
                drop_utt, orig_utt, drop_len / 3600, orig_len / 3600, self.chunk_size
            ))
            logger.info('Used data: {} utterances, ({:.2f} h)'.format(
                self.lens_dict[track], (orig_len-drop_len) / 3600
            ))

        logger.info('Resample pool: {}'.format(list(self.resample_pool.keys())))


    def __len__(self):
        return self.length # can be any number


    def __getitem__(self, idx:int):

        batch = {}
        for track in self.csv_files.keys():
            idx = np.random.randint(self.lens_dict[track])
            wav_info = self.metadata_dict[track].iloc[idx]
            chunk_len = int(wav_info['sr'] * self.chunk_size)

            # slice wav files
            if self.trim_silence: 
                start = np.random.randint(int(wav_info['start']), int(wav_info['end']) - chunk_len + 1)
            else:
                start = np.random.randint(0, int(wav_info['length']) - chunk_len + 1)

            window_start = start
            window_end = start + chunk_len

            if track == 'sfx':

                #print(wav_info['filepath'])
                annots = wav_info['filepath'].split('/')[:-1] + ['annots.csv']
                annots = '/'.join(annots)

                #print(annots)

                annots = pd.read_csv(annots)

                annots = annots[~annots["class"].isin(['speech', 'music']) ]

                annots = annots.drop(columns = ['clip start sample', 'clip end sample', 'clip gain', 'file', 'Unnamed: 0', 'class'])

                annots[["mix start time", "mix end time"]] = annots[["mix start time", "mix end time"]] * 16000

                annots[["mix start time", "mix end time"]] = annots[["mix start time", "mix end time"]].astype(int)

                annots['annotation'] = annots["annotation"].str.rstrip(",").str.replace("_", " ") 
                #print(annots)
                condition_start_inside = (annots["mix start time"] >= window_start) & (annots["mix start time"] <= window_end)  # Starts in window
                condition_end_inside = (annots["mix end time"] >= window_start) & (annots["mix end time"] <= window_end)  # Ends in window
                condition_overlapping = (annots["mix start time"] <= window_start) & (annots["mix end time"] >= window_end)  # Fully covers the window
                condition_fully_inside = (annots["mix start time"] >= window_start) & (annots["mix end time"] <= window_end)  # Fully inside window

                
                annots = annots[condition_start_inside | condition_end_inside | condition_overlapping | condition_fully_inside]
                sentence = annots["annotation"].str.cat(sep=",")
                #print(sentence)

                if not sentence:
                    sentence = 'sfx'

                batch[f'{track}/prompt'] = sentence.lower()

            else:
                batch[f'{track}/prompt'] = track.lower()

            # load file
            x, sr = torchaudio.load(wav_info['filepath'],
                                    frame_offset=start,
                                    num_frames=chunk_len)

            # single channel
            x = x.mean(dim=0, keepdim=True)

            # resample
            if sr != self.sample_rate:
                x = self.resample_pool[sr](x)

            batch[track] = x

        return batch



class DatasetAudioVal(Dataset):
    def __init__(self,
        sample_rate: int,
        tsv_filepath: str,
        chunk_size: float = 5.0,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        self.tsv_filepath = Path(tsv_filepath)
        self.chunk_size = chunk_size # negative for entire sentence
        self.resample_pool = dict()

        # read manifest tsv file
        if self.tsv_filepath.is_file():
            metadata = pd.read_csv(self.tsv_filepath)
            logger.info(f'Manifest filepath: {self.tsv_filepath}')
        else:
            logger.error('No tsv file found in: {}'.format(self.tsv_filepath))

        # audio lengths
        wav_lens = (metadata['end'] - metadata['start']) / metadata['sr']

        # remove wav files that too short
        orig_utt = len(metadata)
        orig_len, drop_utt, drop_len = 0, 0, 0
        drop_rows = []
        for row_idx in range(len(wav_lens)):
            orig_len += wav_lens[row_idx]
            if wav_lens[row_idx] < self.chunk_size:
                drop_rows.append(row_idx)
                drop_utt += 1
                drop_len += wav_lens[row_idx]
            else:
                # prepare julius resample
                sr = int(metadata.at[row_idx, 'sr'])
                if sr not in self.resample_pool.keys():
                    old_sr = sr
                    new_sr = self.sample_rate
                    gcd = math.gcd(old_sr, new_sr)
                    old_sr = old_sr // gcd
                    new_sr = new_sr // gcd
                    self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

        logger.info("Drop {}/{} utts ({:.2f}/{:.2f}h), shorter than {:.2f}s".format(
            drop_utt, orig_utt, drop_len / 3600, orig_len / 3600, self.chunk_size
        ))
        logger.info('Actual data size: {} utterance, ({:.2f} h)'.format(
            orig_utt-drop_utt, (orig_len-drop_len) / 3600
        ))
        logger.info('Resample pool: {}'.format(list(self.resample_pool.keys())))

        self.metadata = metadata.drop(drop_rows)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx:int):

        wav_info = self.metadata.iloc[idx]
        chunk_len = int(wav_info['sr'] * self.chunk_size)
        start = wav_info['start']
        window_start= start

        window_end = start + chunk_len

        annots = wav_info['sfx'].split('/')[:-1] + ['annots.csv']
        annots = '/'.join(annots)

            #print(annots)

        annots = pd.read_csv(annots)

        annots = annots[~annots["class"].isin(['speech', 'music']) ]

        annots = annots.drop(columns = ['clip start sample', 'clip end sample', 'clip gain', 'file', 'Unnamed: 0', 'class'])

        annots[["mix start time", "mix end time"]] = annots[["mix start time", "mix end time"]] * 16000

        annots[["mix start time", "mix end time"]] = annots[["mix start time", "mix end time"]].astype(int)

        annots['annotation'] = annots["annotation"].str.rstrip(",").str.replace("_", " ")
            #print(annots)
        condition_start_inside = (annots["mix start time"] >= window_start) & (annots["mix start time"] <= window_end)  # Starts in window
        condition_end_inside = (annots["mix end time"] >= window_start) & (annots["mix end time"] <= window_end)  # Ends in window
        condition_overlapping = (annots["mix start time"] <= window_start) & (annots["mix end time"] >= window_end)  # Fully covers the window
        condition_fully_inside = (annots["mix start time"] >= window_start) & (annots["mix end time"] <= window_end)  # Fully inside window


        annots = annots[condition_start_inside | condition_end_inside | condition_overlapping | condition_fully_inside]
        sentence = annots["annotation"].str.cat(sep=",")
 
        if not sentence:

            sentence = 'sfx'


        batch = {}

        # Load wav files and resample if needed
        for track in ['mix', 'speech', 'music', 'sfx']:
            x, sr = torchaudio.load(wav_info[track],
                                    frame_offset=start,
                                    num_frames=chunk_len)
            x = x.mean(dim=0, keepdim=True)
            # resample
            if sr != self.sample_rate:
                x = self.resample_pool[sr](x)
            batch[track] = x
            if track != 'sfx':
                batch[f'{track}/prompt'] = track
            else:
                batch[f'{track}/prompt'] = sentence.lower()

        return batch
    


