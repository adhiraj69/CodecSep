"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec


"""
import math

import torch
from torch import nn

from .layers import Snake1d, WNConv1d, WNConvTranspose1d, TransformerSentenceEncoderLayer

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)



class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()

        self.condition_size = condition_size

        self.modules, _ = self.create_film_modules(
            film_meta=film_meta, 
            ancestor_names=[],
        )
        
    def create_film_modules(self, film_meta, ancestor_names):

        modules = {}
       
        # Pre-order traversal of modules
        for module_name, value in film_meta.items():

            if isinstance(value, int):

                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)

                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value, 
                    unique_module_name=unique_module_name,
                )

            elif isinstance(value, dict):

                ancestor_names.append(module_name)
                
                modules[module_name], _ = self.create_film_modules(
                    film_meta=value, 
                    ancestor_names=ancestor_names,
                )

            ancestor_names.pop()

        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):

        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)

        return layer

    def forward(self, conditions):
        
        film_dict = self.calculate_film_data(
            conditions=conditions, 
            modules=self.modules,
        )

        return film_dict

    def calculate_film_data(self, conditions, modules):

        film_data = {}

        # Pre-order traversal of modules
        for module_name, module in modules.items():

            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :,  None]

            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)

        return film_data


def get_film_meta(module):

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.dim #bn1.num_features
            film_meta['beta2'] = module.dim #bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta
    
    return film_meta



class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, has_film: bool = False):
        super().__init__()
        k = 7 # kernal size for the first conv
        pad = ((k - 1) * dilation) // 2 # 2*p - d*(k-1)
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=k, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.has_film = has_film
        self.dim = dim
        ##if self.has_film:
        #    self.bn1 = nn.BatchNorm1d(dim)
        #    self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x, film_dict = None):
        if self.has_film:
            b1 = film_dict['beta1']
            b2 = film_dict['beta2']

            #print('inp',x.shape)
            y = x+ b1
            #print('out',y.shape, b1.shape, self.bn1(x).shape)
            #print(self.block[0])
            y  = self.block[0](y)
            #print(y.shape)
            y= self.block[1](y)
            y = y + b2

            y= self.block[2](y)

            y= self.block[3](y)
        else:
            y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2 # identical in-out channel 
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, has_film: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, has_film = has_film),
            ResidualUnit(dim // 2, dilation=3, has_film = has_film),
            ResidualUnit(dim // 2, dilation=9, has_film= has_film),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )
        self.has_film_ = has_film

    def forward(self, x, film_dict = None):
        if not self.has_film_:
            return self.block(x)
        else:
            for i in range(0,3):
                x= self.block[i](x, film_dict['block'][f'{i}'])
            x = self.block[3](x)
            x= self.block[4](x)
            return x 


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.floor(stride / 2), # ceil() -> floor(), for 16kHz
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)

    
class DACEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 1024,
        has_film: bool= False
    ):
        super().__init__()
        # Create first convolution
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlockd_models that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride, has_film = has_film)]

        # Create last convolution
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        #if not has_film:
        self.block = nn.Sequential(*layers)
        #else:
         #   self.block = layers
        self.enc_dim = d_model
        self.has_film_ = has_film
    def forward(self, x, film_dict = None):
        #print(x.shape)
        if not self.has_film_:
            return self.block(x)
        else:
            #print(self.block[0])
            x = self.block[0](x)
            #print(x.shape)
            N = len(self.block)
            for i in range(1,N-2):
                #print(x.shape)
                x = self.block[i](x, film_dict['block'][f'{i}'])
                #print(x.shape)
            x = self.block[-2](x)    
            x = self.block[-1](x)
            return x 
class DACEncoderTrans(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        att_d_model: int = 512,
        att_nhead: int = 8,
        att_ff: int = 2048,
        att_norm_first: bool = False,
        att_layers: int = 1,
        d_latent: int = 1024,
    ):
        super().__init__()
        # Create first convolution
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlockd_models that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        # Create convolution
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, att_d_model, kernel_size=3, padding=1),
        ]

        # Create attention layers
        layers += [
            TransformerSentenceEncoderLayer(d_model=att_d_model, nhead=att_nhead, 
                                            dim_feedforward=att_ff, norm_first=att_norm_first,
                                            num_layers=att_layers)
        ]

        # Create last convolution
        layers += [
            Snake1d(att_d_model),
            WNConv1d(att_d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)
    

class DACDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 1536,
        strides: list = [8, 8, 4, 2],
        d_latent: int = 1024,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(d_latent, d_model, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks (from HiFi GAN)
        for stride in strides:
            layers += [DecoderBlock(d_model, d_model//2, stride)]
            d_model = d_model // 2

        # Add final conv layer
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DACDecoderTrans(nn.Module):
    def __init__(
        self,
        d_model: int = 1536,
        strides: list = [8, 8, 4, 2],
        d_latent: int = 1024,
        att_d_model: int = 512,
        att_nhead: int = 8,
        att_ff: int = 2048,
        att_norm_first: bool = False,
        att_layers: int = 1,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(d_latent, att_d_model, kernel_size=7, padding=3)]

        # Add attention layer
        layers += [
            TransformerSentenceEncoderLayer(d_model=att_d_model, nhead=att_nhead, 
                                            dim_feedforward=att_ff, norm_first=att_norm_first,
                                            num_layers=att_layers)
        ]

        # Add conv layer
        layers += [
            Snake1d(att_d_model),
            WNConv1d(att_d_model, d_model, kernel_size=7, padding=3)
        ]

        # Add upsampling + MRF blocks (from HiFi GAN)
        for stride in strides:
            layers += [DecoderBlock(d_model, d_model//2, stride)]
            d_model = d_model // 2

        # Add final conv layer
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    


class CodecMixin:
    """Truncated version of DAC CodecMixin
    """
    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L



