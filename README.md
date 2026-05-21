# Official Code for "CodecSep:  Prompt-Driven Universal Sound Separation on Neural Audio Codec Latents"

This repository accompanies our paper [CodecSep:  Prompt-Driven Universal Sound Separation on Neural Audio Codec Latents](https://openreview.net/forum?id=r63GX9hKhC).
[arXiv Link](https://arxiv.org/abs/2509.11717).

The codebase is organized to support experimentation with neural audio codec latent representations through a modular codec interface.

## Enviroment 

`GPUs used: 1 (24GB Nvidia A30)`

`Python Version:  3.11.7`

`PyTorch Version: 2.1.2` 

All experiments were run using `accelerate` with mixed precision support.

`Accelerate Version: 1.6.0`

Setting up conda environment,

```
conda env create -f environment.yml
conda activate gen_audio
```


 [VisQol](https://github.com/google/visqol) Installation:  Read ./install_visqol.md



## Dataset Preparation

We use the following dataset:
- [Divide and Remaster (dnr-v2)](https://zenodo.org/records/6949108)
- [AudioCaps](https://audiocaps.github.io)


### For dnr-v2 dataset,


Default Dataset path: 

`PATH_TO_DNR = 'datasets/dnr_v2/'`

```bash
mkdir manifest

# dnr-v2
python3 prepare/mani_dnr_USS.py --data-dir PATH_TO_DNR

```

### For AudioCaps dataset,

There is no need to generate a manifest for AudioCaps, as we load it directly from [HuggingFace](https://huggingface.co/datasets/OpenSound/AudioCaps)

For evaluation, we create single instances of the test and validation splits with sizes comparable to dnr-v2, and save them in `datasets/audiocaps_test` and `datasets/audiocaps_valid`, respectively. 

```bash
mkdir datasets/audiocaps_test
mkdir datasets/audiocaps_valid
python3 datasets/generate_audiocaps_test_validation_sets.py

```

## Training

**Note:** Please comment out the following line in `src/metrics/__init__.py` **during training**:
```python
from .visqol import VisqolMetric
```
Make sure to **uncomment** it during **evaluation**.


### For training on dnr-v2 dataset, 

```
accelerate launch --config_file config/acc/fp16_gpus1.yaml main_dnr.py --config-name codecsep_dnr +run_config=slurm_codecsep_dnr

```
### For training on AudioCaps dataset, 

```
accelerate launch --config_file config/acc/fp16_gpus1.yaml main_audiocaps.py --config-name codecsep_audiocaps +run_config=slurm_codecsep_audiocaps

```
## Evaluation

Model Checkpoints are not attached (Size > 1 GB  when zipped). 

### For evaluation on dnr-v2 dataset, 

Default model save path:

`PATH_TO_MODEL = 'model-checkpoints/CodecSep_DNR_USS_Weights'`


Generate CodecSep inference outputs on dnr-v2 test:
```
model_dir=PATH_TO_MODEL

nohup python3 eval_dnr_codecsep_inference.py --ret-dir ${model_dir}
```
Generate CodecSep results on dnr-v2 test:
```
nohup python3 eval_dnr_outputs.py --ret-dir ${model_dir} > ${model_dir}/test_dnr.log  2>&1 &
```

### For evaluation on AudioCaps dataset, 

Default model save path:

`PATH_TO_MODEL = 'model-checkpoints/CodecSep_AudioCaps_USS_Weights'`


Generate CodecSep inference outputs on AudioCaps test:
```
model_dir=PATH_TO_MODEL

nohup python3 eval_audiocaps_codecsep_inference.py --ret-dir ${model_dir}
```
Generate CodecSep results on dnr-v2 test:
```
nohup python3 eval_audiocaps_outputs.py --ret-dir ${model_dir} > ${model_dir}/test_dnr.log  2>&1 &
```

## Acknowledgments
The code in this project is adapted or modifed from the following projects:
- [SDCodec](https://github.com/XiaoyuBIE1994/SDCodec) [MIT License]
- [AudioCraft](https://github.com/facebookresearch/audiocraft) [MIT License]
- [DAC](https://github.com/descriptinc/descript-audio-codec) [MIT License]



