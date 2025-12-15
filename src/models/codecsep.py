"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec
and Codecformer: https://github.com/Yip-Jia-Qi/codecformer

"""
import math
import numpy as np
from typing import List, Union, Tuple
from typing import Optional
import random
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
import dac
from ..modules.base_dac_F import get_film_meta, FiLM
import laion_clap
from ..modules import CodecMixin
from .. import modules

logger = get_logger(__name__)
                    
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02) # default kaiming_uniform_
        nn.init.constant_(m.bias, 0) # default uniform_




class PositionalEncoding(nn.Module):


    def __init__(self, input_size, max_len=2500):
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd channels (got channels={input_size})"
            )
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        return self.pe[:, : x.size(1)].clone().detach()


class simpleSeparator2(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels, has_film = False):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))
        self.pos_enc = PositionalEncoding(256)
        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        #self.output_gate = nn.Sequential(
        #    nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        #)
        self.dim = channels
        self.has_film = has_film


    def forward(self,x, film_dict = None):
        
        x = self.ch_down(x)
        #[B,N,L]
        x = x.permute(0,2,1)
        #[B,L,N]
        x = x + self.pos_enc(x)
        x_b = self.block(x, film_dict = film_dict['block'])
        #[B,L,N]
        x_b = x_b.permute(0,2,1)
        #[B,N,L]
        x = self.ch_up(x_b)

        if self.has_film:
            b1 = film_dict['beta1']
            b2 = film_dict['beta2']

        if self.has_film:
            #print(x.shape, b1.shape)
            x = x.permute(0,2,1) + b1.permute(0,2,1)
            x = x.permute(0,2,1)
        B, N, L = x.shape
        masks = self.masker(x)
        
        #[B,N*num_spks,L]
        masks = masks.view(B*self.num_spks,-1,L)
        #b2 = film_dict['beta2']
        if self.has_film:

            masks = masks.permute(0,2,1) + b2.permute(0,2,1)
            masks = masks.permute(0,2,1)
        #[B*num_spks, N, L]
        x = self.output(masks) # * self.output_gate(masks)
        #x = self.activation(x)

        #[B*num_spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        
        # [B, spks, N, L]
        x = x.transpose(0,1)
        # [spks, B, N, L]

        return x






class TransformerEncoderLayerFiLM(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim,batch_first = False, dropout=0.1, has_film = False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads,batch_first = batch_first, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.has_film = has_film
        self.dim = model_dim
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.has_film = has_film

    def forward(self, x, film_dict = None):
        if self.has_film:
            b1 = film_dict['beta1']
            b2 = film_dict['beta2']
            #print(b1.shape)

        #if self.has_film:
            #print(x.shape, b1.shape)
            #x = x + b1.squeeze(-1)

        attn_output, _ = self.self_attn(x, x, x)
        #print(attn_output.shape)
        if self.has_film:
            x = x + self.dropout(attn_output) # + b1.squeeze(-1)
        else:
            x = x + self.dropout(attn_output)
        
        x = self.norm1(x) #+ b1.squeeze(-1)

        #print(x.shape)
        if self.has_film:
            x = x + b1.permute(0,2,1)

        ffn_output = self.ffn(x)
        if self.has_film:
            x = x + self.dropout(ffn_output) #+ b2.squeeze(-1)
        else:
            x = x + self.dropout(ffn_output)

        x = self.norm2(x) 

        if self.has_film:
            x = x + b2.permute(0,2,1)
        
        return x



class TransformerEncoderWithFiLM(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, film_condition_size,batch_first = False, dropout=0.1):
        super().__init__()
        
        
        #self.input = TransformerEncoderLayerFiLM(model_dim=model_dim,
        #        num_heads=num_heads,
        #        ff_dim=ff_dim,
        #        dropout=dropout, has_film = False, batch_first = batch_first )

        self.layers = nn.ModuleList([
            TransformerEncoderLayerFiLM(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                has_film = True,
                batch_first = batch_first,
            ) for _ in range(num_layers)
        ])

        #self.output = TransformerEncoderLayerFiLM(model_dim=model_dim,
        #        num_heads=num_heads,
        #        ff_dim=ff_dim,
        #        dropout=dropout, batch_first = batch_first, has_film = False )


        
    def forward(self, x, film_dict = None):
        
        #x = self.input(x)
        for i in range( len(self.layers)):
            x = self.layers[i]( x, film_dict['layers'][f'{i}'])
        
        #x = self.output(x)
        return x









class CodecSep(nn.Module, CodecMixin):

    def __init__(
        self,
        sample_rate: int,
        latent_dim: int = None,
        tracks: List[str] = ['speech', 'music', 'sfx'],
        enc_params: DictConfig = {'name': 'DACEncoder'},
        dec_params: DictConfig = {'name': 'DACDecoder'},

        
        transformer_params: DictConfig = {'name': 'Transformer'},
        separator_params: DictConfig = {'name': 'Separator'},

        pretrain: dict = {},
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.tracks = tracks
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.transformer_params = transformer_params
        self.separator_params = separator_params
        self.pretrain = pretrain

        self.track2idx = {'speech':0, 'music':1, 'sfx': 2}
        if latent_dim is None:
            latent_dim = self.enc_params.d_model * (2 ** len(self.enc_params.strides))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(self.enc_params.strides)

        enc_net = getattr(modules, self.enc_params.pop('name'))
        dec_net = getattr(modules, self.dec_params.pop('name'))
        self.encoder = enc_net(**self.enc_params)
        self.decoder = dec_net(**self.dec_params)
        self.separator_params.pop('name')
        self.transformer_params.pop('name')
        # Define separator

        self.transformer_encoder = TransformerEncoderWithFiLM(num_layers= self.transformer_params.num_layers, 
                                        model_dim = self.transformer_params.d_model,
                                        num_heads= self.transformer_params.nhead,
                                        ff_dim = self.transformer_params.dim_feedforward,
                                        film_condition_size = self.transformer_params.d_model,
                                        dropout=self.transformer_params.dropout,
                                        batch_first = self.transformer_params.batch_first)         
       


        self.text_encoder = laion_clap.CLAP_Module( enable_fusion=False, amodel= 'HTSAT-base')




        self.separator = simpleSeparator2(num_spks = 1,  has_film = True, block = self.transformer_encoder, channels = self.separator_params.channels, block_channels = self.separator_params.block_channels)  #(modules, self.quant_params.pop('name'))
        #self.quantizer = separator_net(tracks=self.tracks, **self.separator_params)

        film_meta = get_film_meta(self.separator)
        #print(film_meta)
        self.film = FiLM(film_meta, condition_size = 512)



        # Init
        #self.apply(init_weights)
        self.delay = self.get_delay()

        # Load pretrained
        load_pretrained = self.pretrain.get('load_pretrained', False)
        if load_pretrained:
            pretrained_dict = torch.load(load_pretrained)
            hyparam = pretrained_dict['metadata']['kwargs']
            ignore_modules = self.pretrain.get('ignore_modules', [])
            freeze_modues = self.pretrain.get('freeze_modules', [])
            
            is_match = self._check_hyparam(hyparam)
            if is_match:
                self._load_pretrained(pretrained_dict['state_dict'], ignore_modules)
                self._freeze(freeze_modues)
                logger.info('Pretrain models load success from {}'.format(load_pretrained))
                logger.info('-> modules ignored: {}'.format(ignore_modules))
                logger.info('-> modules freezed: {}'.format(freeze_modues))
            else:
                logger.info(f'Pretrain param do not match model, load pretrained failed...')
                logger.info('Pretrain params:')
                logger.info(hyparam)
                logger.info('Model params:')
                logger.info('Encoder: {}'.format(self.enc_params))
                logger.info('Decoder: {}'.format(self.dec_params))

    @property
    def device(self):
        """Gets the device the model is on by looking at the device of
        the first parameter. May not be valid if model is split across
        multiple devices.
        """
        return list(self.parameters())[0].device


    def _check_hyparam(self, hyparam):
        return (hyparam['encoder_dim'] == self.enc_params['d_model']) \
            and (hyparam['encoder_rates'] == self.enc_params['strides']) \
            and (hyparam['decoder_dim'] == self.dec_params['d_model']) \
            and (hyparam['sample_rate'] == self.sample_rate)


    def _freeze(self, freeze_modues=[]):
        for module in freeze_modues:
            child = getattr(self, module)
            for param in child.parameters():
                param.requires_grad = False


    def preprocess(self, audio_data, sample_rate) -> torch.Tensor:
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))

        return audio_data


    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
 
        return self.encoder(audio_data)
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        return self.decoder(z)
    

    def forward(
        self,
        batch: Optional[dict],
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
 
        
        # mix by masking 
        audio_data = batch['mix']
        bs, _, length = audio_data.shape
        valid_tracks = batch['valid_tracks']
        mask = [1 if t in valid_tracks else 0 for t in self.tracks]
        mask = torch.tensor(mask, device=audio_data.device)
        conditions = self.tracks
     
        #conditions = ['speech', 'music'] + batch['prompt']
        #print(conditions)

        #print(batch['speech/prompt'], batch['music/prompt'])
        text_embed = { t : self.text_encoder.get_text_embedding(batch[f'{t}/prompt'], use_tensor=True).detach() for t in conditions}

        #print(text_embed['speech'].shape)
        film_dicts = {f'{t}': self.film(text_embed[t].to(self.device)) for t in self.tracks }

        # preprocess, zero-padding to proper length
        audio_data = self.preprocess(audio_data, sample_rate)

        # encoder
        feats = self.encode(audio_data)
        sep_masks = []
        for i, track in enumerate(self.tracks):

                    est_mask = self.separator(feats, film_dict = film_dicts[f'{track}'])
                    sep_masks.append( est_mask.squeeze(0))
                    #print(f'mask {i}', est_mask.shape)
        est_mask = torch.stack(sep_masks)

        #print('est_mask',est_mask.shape)
        feats = torch.stack([feats] * len(self.tracks))
        #print('fearture',feats.shape)
        recon = feats * est_mask


        mix_ = recon.sum(0)
        #print('recon', recon.shape, mix_.shape)
        audio_recons = [self.decode(recon[self.track2idx[t]])[:,:,:audio_data.shape[-1]] for t in valid_tracks]
        mixture = self.decode(mix_)[:,:,:audio_data.shape[-1]]
        #print(audio_data.shape, audio_recons[0].shape)
        audio_recon = torch.stack([mixture] +  audio_recons, dim = 1)


        batch['recon'] = audio_recon[..., :length]
        batch['length'] = length

        return batch



    def evaluate(
            self,
            input_audio, #: torch.Tensor,
            #prompt : list[str] = ['sfx'],
            sample_rate: int = None,
            n_quantizers: int = None,
            output_tracks: list[str] = ['mix'],
    ) -> torch.Tensor:

        assert all((t in self.tracks) or (t=='mix') for t in output_tracks); \
        "output tracks {} not included in model tracks {}".format(output_tracks, self.tracks)
        conditions = self.tracks

        input_audio, prompt = input_audio[0] , input_audio[1]
        conditions = self.tracks
        #print(input_audio)
        conditions = prompt
        #print(conditions)

        text_embed = { t: self.text_encoder.get_text_embedding( conditions[self.track2idx[t]], use_tensor=True).detach() for t in self.tracks}

        #print(text_embed['music'].shape)
        film_dicts = {f'{t}': self.film(text_embed[t].to(self.device)) for t in self.tracks }



        bs, _, length = input_audio.shape
        audio_data = self.preprocess(input_audio, sample_rate) # (B, 1, T)



        #print(film_dicts.keys())
        # encoder
        feats = self.encode(audio_data)

        sep_masks = []
        for i, track in enumerate(self.tracks):
                    #print(track)
                    est_mask = self.separator(feats, film_dict = film_dicts[f'{track}'])
                    #print(f'{track}', est_mask.shape)
                    sep_masks.append(est_mask.squeeze(0))

        est_mask = torch.stack(sep_masks)
        #print('MASK_ESTIMATE', est_mask.shape)
        #est_mask = self.separator(feats)
        feats = torch.stack([feats] * len(self.tracks))

        recon = feats * est_mask

        #print('reconstr', recon.shape)
        mix_ = recon.sum(0)

        #print(mix_.shape)        
        list_out = []

        for track in output_tracks:
            if track == 'mix':

                x_out =self.decode(mix_)[:,:,:audio_data.shape[-1]]
                #print('MIx audio', x_out.shape)
                list_out.append(x_out)
            else:
                x_out = self.decode(recon[self.track2idx[track]])[:, :, :audio_data.shape[-1]]
                #print(f'{track}', x_out.shape)
                list_out.append(x_out)



        output_audio = torch.cat(list_out, dim=1)[...,:length]

        return output_audio


    @torch.no_grad()
    def evaluate_quantized(
        self,
        input_audio_and_prompt,
        sdcodec,                         # an SDCodec instance with a trained quantizer
        sample_rate: int = None,
        n_quantizers: int = None,
        output_tracks: list[str] = ['mix'],
    ) -> torch.Tensor:
        """
        Evaluate CodecSep using SDCodec's quantized representation as input to the separator,
        and re-quantize source estimates before decoding to audio.

        Args:
            input_audio_and_prompt: Tuple[Tensor[B,1,T], Dict/list-like prompts]]
                Same contract as `evaluate`: (audio_tensor, prompt_dict_or_list)
                Where prompts are in order of self.tracks or accessible by names.
            sdcodec: SDCodec instance (must expose .quantize(feats, track, n_quantizers))
            sample_rate: optional; defaults to self.sample_rate
            n_quantizers: optional; number of codebooks to use (pass through to SDCodec)
            output_tracks: e.g., ['mix', 'speech', 'music', 'sfx']
        Returns:
            Tensor[B, K, 1, T] (K = len(output_tracks))
        """
        # --- unpack & checks
        assert all((t in self.tracks) or (t == 'mix') for t in output_tracks), \
            f"output tracks {output_tracks} not included in model tracks {self.tracks}"

        input_audio, prompt = input_audio_and_prompt
        B, _, length = input_audio.shape

        # text → FiLM
        if isinstance(prompt, dict):
            # expect keys that match track names
            get_txt = lambda t: prompt[t]
        else:
            # assume list/tuple aligned with self.tracks
            get_txt = lambda t: prompt[self.track2idx[t]]

        text_embed = {
            t: self.text_encoder.get_text_embedding(get_txt(t), use_tensor=True).detach()
            for t in self.tracks
        }
        film_dicts = {t: self.film(text_embed[t].to(self.device)) for t in self.tracks}

        # --- preprocess & encode (continuous)
        audio_data = self.preprocess(input_audio, sample_rate)  # (B,1,T)
        feats = self.encode(audio_data)                         # (B,D,T)

        # --- quantize with SDCodec (per-track) and build quantized mix
        # Note: We quantize the SAME feats through each track-specific RVQ, then sum (SDCodec-style).
        dict_qz = {}
        for t in self.tracks:
            with torch.cuda.amp.autocast(enabled=False):
                feats32 = feats.to(dtype=torch.float32)
                qz, _, _, _, _ = sdcodec.quantize(feats32, track=t, n_quantizers=n_quantizers)  # (B,D,T)
                dict_qz[t] = qz

        # Quantized mix latent to feed separator
        feats_q_mix = torch.stack([dict_qz[t] for t in self.tracks], dim=0).sum(dim=0)    # (B,D,T)

        # --- separation on quantized latent
        sep_masks = []
        for t in self.tracks:
            est_mask = self.separator(feats_q_mix, film_dict=film_dicts[t])  # [spks,B,D,T], spks=1
            sep_masks.append(est_mask.squeeze(0))                             # (B,D,T)
        est_mask = torch.stack(sep_masks, dim=0)                              # (Ttracks,B,D,T)

        # Reconstruct source latents from quantized mix latent
        feats_q_mix_stack = torch.stack([feats_q_mix] * len(self.tracks), dim=0)  # (Ttracks,B,D,T)
        recon_latents = feats_q_mix_stack * est_mask                               # (Ttracks,B,D,T)

        # --- re-quantize each source latent with the corresponding SDCodec track quantizer
        rq_latents = {}
        for t in self.tracks:
            recon_t = recon_latents[self.track2idx[t]]                             # (B,D,T)
            with torch.cuda.amp.autocast(enabled=False):
                recon_t32 = recon_t.to(dtype=torch.float32)
                rqz_t, _, _, _, _ = sdcodec.quantize(recon_t32, track=t, n_quantizers=n_quantizers)
                rq_latents[t] = rqz_t

        # --- build outputs
        list_out = []
        for track in output_tracks:
            if track == 'mix':
                # sum re-quantized sources for mix, then decode
                z_mix_out = torch.stack([rq_latents[t] for t in self.tracks], dim=0).sum(dim=0)
                x_out = self.decode(z_mix_out)[:, :, :audio_data.shape[-1]]         # (B,1,T)
            else:
                z_t = rq_latents[track]
                x_out = self.decode(z_t)[:, :, :audio_data.shape[-1]]               # (B,1,T)
            list_out.append(x_out)

        output_audio = torch.cat(list_out, dim=1)[..., :length]  # (B, K, 1, T_trim)
        return output_audio


