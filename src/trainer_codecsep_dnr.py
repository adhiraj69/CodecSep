"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec


"""
import sys
import shutil
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict, defaultdict
import random
from einops import rearrange


from src.datasets.audio_dataset_dnr import DatasetAudioTrain, DatasetAudioVal

import torch

print(torch.__file__)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.distributions import Categorical

from src import datasets, models, optim, utils
from src.metrics import SingleSrcNegSDR


from src.models.codecsep import CodecSep
from src.models.sdcodec import SDCodec
logger = get_logger(__name__) # avoid douplicated print, params defined by Hydra
accelerator = Accelerator(project_dir=HydraConfig.get().runtime.output_dir,
                          step_scheduler_with_optimizer=False,
                          log_with="tensorboard")

class Trainer(object):
    def __init__(self, cfg: DictConfig):

        # Init
        self.cfg = cfg
        
        OmegaConf.set_struct(self.cfg, False) # enable config.pop()
        self.project_dir = Path(accelerator.project_dir)
        self.device = accelerator.device
        logger.info('Init Trainer')

        # Fix random
        seed = self.cfg.training.get('seed', False)
        if seed:
            set_seed(seed)

        # Backup code, only on main process
        if self.cfg.backup_code and accelerator.is_main_process:
            back_dir = self.project_dir / 'backup_src'
            logger.info(f'Backup code at: {back_dir}')
            cwd = HydraConfig.get().runtime.cwd
            src_dir = Path(cwd) / 'src'
            if back_dir.exists():
                shutil.rmtree(back_dir)
            shutil.copytree(src=src_dir, dst=back_dir)

        # Checkpoint
        self.ckpt_best = self.project_dir / 'ckpt_best'
        self.ckptdir = self.project_dir / 'checkpoints'
        self.ckptdir.mkdir(exist_ok=True)
        self.ckpt_final = self.project_dir / 'ckpt_final'
        self.ckpt_final.mkdir(exist_ok=True)
        # Check resume
        if self.cfg.resume:
            resume_dir = self.cfg.get('resume_dir', None)
            if resume_dir is None:
                logger.info(f'No resume_dir provided, try to resume from best ckpt directory...')
                self.ckpt_resume = self.ckpt_best
            else:
                self.ckpt_resume = self.project_dir / resume_dir
            if not self.ckpt_best.is_dir():
                self.cfg.resume = False
                logger.info(f'Resume FAILED, no ckpt dir at: {self.ckpt_best}')

        # Tensorboard tracker
        accelerator.init_trackers(project_name='tb')
        self.tracker = accelerator.get_tracker("tensorboard")
        logger.info('Tracker backend: tensorboard')
        
        # Prepare dataset
        self.sr = self.cfg.sampling_rate
        logger.info('=====> Training dataloader')
        self.train_loader = self._get_data(self.cfg.dataset.trainset_cfg, self.cfg.training.dataloader, 
                                           is_train=True, sample_rate=self.sr)
        logger.info('=====> Validation dataloader')
        self.val_loader = self._get_data(self.cfg.dataset.valset_cfg, self.cfg.training.dataloader, 
                                         is_train=False, sample_rate=self.sr)
        logger.info('=====> Test dataloader')
        self.test_loader = self._get_data(self.cfg.dataset.testset_cfg, self.cfg.training.dataloader, 
                                          is_train=False, sample_rate=self.sr)

        # Prepare generator
        model_name = self.cfg.model.codecsep_params.pop('name')
        optim_name = self.cfg.training.optimizer.pop('name')
        scheduler_name = self.cfg.training.scheduler.pop('name')
        self.model = self._get_model(model_name, self.cfg.model.codecsep_params, self.sr)
        self.optimizer_g = self._get_optim(self.model.parameters(), optim_name, self.cfg.training.optimizer)
        self.scheduler_g = self._get_scheduler(self.optimizer_g, scheduler_name, self.cfg.training.scheduler)
        logger.info('=====> Model Best Checkpoint Loading')


        self.model.load_state_dict(torch.load('SD_Codec_weights/ckpt_final/ckpt_model_final.pth'), strict = False)
        logger.info('=====> Model Best Checkpoint Loaded !!!')
       


        self.use_codec_loss = self.cfg.use_codec_loss

        self.model._freeze(['encoder', 'decoder'])

        if self.use_codec_loss:
            print('======> Loading Codec Model for cSI-SDR loss')
            codec_name = self.cfg.model.codec_params.pop('name')
            codec_cfg = self.cfg.model.codec_params

            self.codec = self._get_model(codec_name, codec_cfg, self.sr)
            logger.info('=====> Model Best Checkpoint Loading')
            self.codec.load_state_dict(torch.load('SD_Codec_weights/ckpt_final/ckpt_model_final.pth'), strict = False)
            logger.info('=====> Model Best Checkpoint Loaded !!!')
            self.codec._freeze(['encoder', 'decoder', 'quantizer'])

            self.codec.eval()

        # Prepare training recording
        self.tm = utils.TrainMonitor()

        # Accelerator preparation
        self._acc_prepare()

        # Define the loss function
        self._metric_prepare(self.cfg.training.loss)
        self.lambdas = self.cfg.training.loss.lambdas

        # Define the audio transform function
        self._transform_prepare(self.cfg.training.transform)
        # Resume
        if self.cfg.resume:
            logger.info(f'Resume training from: {self.ckpt_resume}')
            accelerator.load_state(self.ckpt_resume)
            self.tm.nb_step += 1
            logger.info(f'Training re-start from iter: {self.tm.nb_step}')
        else:
            logger.info(f'Experiment workdir: {self.project_dir}')
            logger.info(f'num_processes: {accelerator.num_processes}')
            logger.info(f'batch size per gpu for train: {self.cfg.training.dataloader.train_bs}')
            logger.info(f'batch size per gpu for validation: {self.cfg.training.dataloader.eval_bs}')
            logger.info(f'mixed_precision: {accelerator.mixed_precision}')
            # logger.info(OmegaConf.to_yaml(self.cfg))
            logger.info('Trainer init finish')

        # Basic info
        self.tracks = self.cfg.model.codecsep_params.tracks # ['speech', 'music', 'sfx']
        self.eval_tracks = ['mix_rec'] + [f'{t}_rec' for t in self.tracks] + [f'{t}_sep_mask' for t in self.tracks]
        logger.info('Used audio tracks: {}'.format(self.tracks))
        logger.info('Eval audio tracks: {}'.format(self.eval_tracks))
        logger.info(f'Target sampling rate: {self.sr} Hz')
        logger.info(f'Random seed: {seed}')


    def _get_data(self, dataset_cfg, dataloader_cfg, is_train=True, sample_rate=44100):
        
        num_workers = dataloader_cfg.num_workers
        
        if is_train:
            batch_size = dataloader_cfg.train_bs
            shuffle = True
            drop_last = True
            data_class =DatasetAudioTrain #getattr(datasets, f'DatasetAudioTrain')
        else:
            batch_size = dataloader_cfg.eval_bs
            shuffle = False
            drop_last = False
            data_class = DatasetAudioVal #Wgetattr(datasets, f'DatasetAudioVal')
        
        dataset = data_class(sample_rate=sample_rate, **dataset_cfg)
        dataloader = DataLoader(dataset=dataset, 
                                batch_size=batch_size, num_workers=num_workers,
                                shuffle=shuffle, drop_last=drop_last)
        return dataloader


    def _get_model(self, model_name, model_cfg, sample_rate=44100):
        logger.info(f"Model: {model_name}")

        if model_name == 'CodecSep':
            model = CodecSep(sample_rate, **model_cfg)
    
        else:     
            net_class = getattr(models, f'{model_name}')
        
            model = net_class(sample_rate=sample_rate, **model_cfg)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f'Total params: {total_params:.2f} Mb')
        total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f'Total trainable params: {total_train_params:.2f} Mb')
        return model


    def _get_optim(self, params, optim_name, optim_cfg):
        logger.info(f"Optimizer: {optim_name}")
        optim_class = getattr(torch.optim, optim_name)
        optimizer = optim_class(filter(lambda p: p.requires_grad, params), **optim_cfg)
        return optimizer


    def _get_scheduler(self, optimizer, scheduler_name, scheduler_cfg):
        logger.info(f"Scheduler: {scheduler_name}")
        sche_class = getattr(optim, scheduler_name)
        scheduler = sche_class(optimizer, **scheduler_cfg)
        return scheduler


    def _metric_prepare(self, loss_cfg):

        self.eval_sisdr = SingleSrcNegSDR(sdr_type='sisdr')


    def _transform_prepare(self, transform_cfg):
        self.volume_norm = utils.VolumeNorm(sample_rate=self.sr)
        self.sep_norm = utils.WavSepMagNorm()
        self.peak_norm = utils.db_to_gain(transform_cfg.peak_norm_db)


    def _acc_prepare(self):
        self.model = accelerator.prepare(self.model)
        if self.use_codec_loss:
            self.codec = accelerator.prepare(self.codec)

        self.optimizer_g = accelerator.prepare(self.optimizer_g)
        self.scheduler_g = accelerator.prepare(self.scheduler_g)
        self.train_loader = accelerator.prepare(self.train_loader)
        self.val_loader = accelerator.prepare(self.val_loader)
        self.test_loader = accelerator.prepare(self.test_loader)
        accelerator.register_for_checkpointing(self.tm)
        self.model_eval_func = self.model.module.evaluate if accelerator.use_distributed \
               else self.model.evaluate
        logger.info('{} iterations per epoch'.format(len(self.train_loader)))


    def _data_transform(self, batch, transform_cfg, nb_step=-1, is_eval=False):

        # re-build data
        if is_eval:
            batch['valid_tracks'] = self.tracks
            norm_var = 0
        else:
            # random drop 0-2 tracks
            dist = Categorical(probs=torch.tensor(transform_cfg.random_num_sources))
            num_sources = dist.sample() + 1
            batch['valid_tracks'] = random.sample(self.tracks, num_sources)
            norm_var = transform_cfg.lufs_norm_db['var']

        batch['in_sources'] = len(batch['valid_tracks'])
        # build mix
        mix_max_peak = torch.zeros_like(batch['speech'])[...,:1] # (bs, C, 1)
        for track in batch['valid_tracks']:
            # volume norm
            batch[track] = self.volume_norm(signal=batch[track],
                                            target_loudness=transform_cfg.lufs_norm_db[track],
                                            var=norm_var)
            # peak value
            peak = batch[track].abs().max(dim=-1, keepdims=True)[0]
            mix_max_peak = torch.maximum(peak, mix_max_peak)

        # peak norm
        peak_gain = torch.ones_like(mix_max_peak) # (bs, C, 1)
        peak_gain[mix_max_peak > self.peak_norm] = self.peak_norm / mix_max_peak[mix_max_peak > self.peak_norm]
        batch['mix'] = torch.zeros_like(batch['speech'])
        for track in batch['valid_tracks']:
            batch[track] *= peak_gain
            batch['mix'] += batch[track]
        # mix volum norm
        batch['mix'], mix_gain = self.volume_norm(signal=batch['mix'],
                                                  target_loudness=transform_cfg.lufs_norm_db['mix'],
                                                  var=norm_var,
                                                  return_gain=True)
        
        # norm each track
        for track in batch['valid_tracks']:
            batch[track] *= mix_gain[:, None, None]


        if self.use_codec_loss:
            for track in batch['valid_tracks']:
                batch[f'{track}/codec'] = self.codec.evaluate(batch[track], output_tracks=[track] )

        # random swap tracks
        batch['random_swap'] = (not is_eval) and (random.random() < transform_cfg.random_swap_prob)
        if batch['random_swap']:
            bs = batch['mix'].shape[0]
            mix_ref = torch.zeros_like(batch['mix'])
            batch['shuffle_list'] = {}
            for track in self.tracks:
                shuffle_list = list(range(bs))
                random.shuffle(shuffle_list)
                batch['shuffle_list'][track] = shuffle_list
                if track in batch['valid_tracks']:
                    mix_ref += batch[track][shuffle_list]
        else:
            mix_ref = batch['mix'].clone()




        batch['ref'] = torch.stack([mix_ref]+[batch[t].clone() for t in batch['valid_tracks']], dim=1) # (B, K, C, T)

        if self.use_codec_loss:
            mix_ref_codec = self.codec.evaluate(mix_ref, output_tracks=['mix'])
            batch['ref_codec'] =  torch.stack([mix_ref_codec]+[batch[f'{t}/codec'].clone() for t in batch['valid_tracks']], dim=1) # (B, K, C, T)
        return batch


    def _print_logs(self, log_dict, title='Train', nb_step=0, use_tracker=True):
        msg = f"{title} iter {nb_step:d}"
        
        if title == 'Train':
            for k, v in log_dict.items():
                k = '/'.join(k.split('/')[1:])
                if k == 'lr':
                    msg += f' {k}: {v:.8f}'
                else:
                    msg += f' {k}: {v:.2f}'
            logger.info(msg)
        else:
            logger.info(msg)
            for c in self.eval_tracks:
                msg = f"--> {c}:"
                select_keys = filter(lambda k: k.split('/')[0] == f'eval_{c}'
                                     or k.split('/')[0] == f'test_{c}', log_dict.keys())
                for k in select_keys:
                    v = log_dict[k]
                    k = '/'.join(k.split('/')[1:])
                    msg += ' {}: {:.2f}'.format(k, v)
                logger.info(msg)

        if use_tracker:
            self.tracker.log(log_dict, step=nb_step) # tracker automatically discard plt and audio


    def run(self):
        total_steps = self.cfg.training.total_steps
        print_steps = self.cfg.training.print_steps
        eval_steps = self.cfg.training.eval_steps
        vis_steps = self.cfg.training.vis_steps
        test_steps = self.cfg.training.test_steps
        early_stop = self.cfg.training.early_stop
        grad_clip = self.cfg.training.grad_clip
        save_iters = self.cfg.training.save_iters

        self.model.train()
        if self.use_codec_loss:
            self.codec.eval() 
        self.model._freeze(['encoder', 'decoder'])

    


        logger.info('Training...')
        while self.tm.nb_step <= total_steps:
            for batch in self.train_loader:
                
                # data transform and augmentation
                batch = self._data_transform(batch, self.cfg.training.transform, self.tm.nb_step, is_eval = True)

                # train one step
                with accelerator.autocast():
                    train_log_dict = self.train_one_step(batch, grad_clip)

                # print log
                if self.tm.nb_step % print_steps == 0:
                    self._print_logs(train_log_dict, title='Train', nb_step=self.tm.nb_step)

                # eval
                if self.tm.nb_step % eval_steps == 0:
                    val_log_dict = self.run_eval()
                    self._print_logs(val_log_dict, title='Validation', nb_step=self.tm.nb_step)

                    # save best val
                    if val_log_dict['val'] < self.tm.best_eval:
                        self.tm.best_eval = val_log_dict['val']
                        self.tm.best_step = self.tm.nb_step
                        logger.info("\t-->Validation improved!!! Save best!!!")
                        accelerator.save_state(output_dir=self.ckpt_best, safe_serialization=False) # otherwise can't reload correctly
                    # early stop
                    else:
                        self.tm.early_stop += 1
                        if self.tm.early_stop >= early_stop:
                            logger.info(f"\t--> Validation no imporved for {early_stop} times")
                            logger.info(f"\t--> Training finished by early stop")
                            logger.info(f"\t--> Best model saved at iter: {self.tm.best_step}")
                            logger.info(f"\t--> Final test, load best ckpt")
                            accelerator.load_state(self.ckpt_best)
                            # save state and end
                            unwrapped_model = accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.ckpt_final / 'ckpt_model_final.pth')

                            logger.info(f"\t--> Final ckpt saved in {self.ckpt_final}")
                            # final test
                            test_log_dict = self.run_test()
                            self._print_logs(test_log_dict, title='Final Test', nb_step=self.tm.best_step, use_tracker=False)
                            accelerator.end_training()
                            return
                
                # save model
                if self.tm.nb_step in save_iters:
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), self.ckptdir / 'ckpt_model_iter{}.pth'.format(self.tm.nb_step))

                    logger.info('\t--> Checkpoints saved for iteration: {}'.format(self.tm.nb_step))

                # vis
                if self.tm.nb_step % vis_steps == 0:
                    self.run_vis(nb_step=self.tm.nb_step)

                # test set
                if self.tm.nb_step % test_steps == 0:
                    test_log_dict = self.run_test()
                    self._print_logs(test_log_dict, title='Test', nb_step=self.tm.nb_step)

                self.tm.nb_step += 1

                # training end due to maximum train iters
                if self.tm.nb_step > total_steps: 
                    logger.info(f"\t--> Training finished by reaching max iterations")
                    logger.info(f"\t--> Best model saved at iter: {self.tm.best_step}")
                    logger.info(f"\t--> Final test, load best ckpt")
                    accelerator.load_state(self.ckpt_best)
                    # save state and end
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), self.ckpt_final / 'ckpt_model_final.pth')

                    logger.info(f"\t--> Final ckpt saved in {self.ckpt_final}")
                    # final test
                    test_log_dict = self.run_test()
                    self._print_logs(test_log_dict, title='Final Test', nb_step=self.tm.best_step, use_tracker=False)
                    accelerator.end_training()
                    return

                # breakpoint()


    def train_one_step(self, batch, grad_clip):
        # Forward, AMP automatically set by Accelerator
        batch = self.model(batch)
        #print(batch['recon'].shape)
        recon = batch['recon']
        #print(recon.shape)
        ref = batch['ref']

        # Reshape in/out
        audio_recon = rearrange(recon, 'b k c t -> (b k) c t')
        audio_ref = rearrange(ref, 'b k c t -> (b k) c t')

        loss = self.eval_sisdr(est = audio_recon, ref= audio_ref)

        if self.use_codec_loss:

            ref_codec = batch['ref_codec']

            audio_ref_codec = rearrange(ref_codec, 'b k c t -> (b k) c t')

            loss_codec = self.eval_sisdr(est= audio_recon, ref=audio_ref_codec)

            loss += loss_codec

            loss /=2


        # debugging nan error
        if loss.isnan().any():
            logger.error('Nan detect, debugging...')
            ckpt_debug = self.project_dir / 'ckpt_debug'
            ckpt_debug.mkdir(exist_ok=True)
            data_debug = ckpt_debug / f'batch_data_{accelerator.process_index}.pth'
            accelerator.save_state(output_dir=ckpt_debug, safe_serialization=False)
            torch.save(batch, data_debug)
            logger.info(f"\t--> Debug state saved in {ckpt_debug}")
            accelerator.wait_for_everyone()
            accelerator.end_training()
            sys.exist()
            return
            # breakpoint()
            

        # Generator gradient descent
        self.optimizer_g.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm_g = accelerator.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
        self.optimizer_g.step()

        # Mean reduce across all GPUs
        log_dict = OrderedDict()
        log_dict['train/lr'] = self.scheduler_g.get_last_lr()[0]

        log_dict['train/grad_norm_g'] = accelerator.reduce(grad_norm_g, reduction="mean").item()

        log_dict['train/sisdr_loss'] = accelerator.reduce(loss, reduction="mean").item()

        return log_dict


    @torch.no_grad()
    def run_eval(self):
        """Distributed evaluation
        for inputs, targets in validation_dataloader:
            predictions = model(inputs)
            # Gather all predictions and targets
            all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
            # Example of use with a *Datasets.Metric*
            metric.add_batch(all_predictions, all_targets)
        """
        self.model.eval()

        am_sisdr = {k: utils.AverageMeter() for k in self.eval_tracks}

        
        for batch in self.val_loader:
            
            # data transform and augmentation
            batch = self._data_transform(batch, self.cfg.training.transform, self.tm.nb_step, is_eval=True)

            # Forward
            batch = self.model(batch)

            # Distributed evaluation
            all_recon = accelerator.gather_for_metrics(batch['recon']) # valide for nested list/tuple/dict
            all_ref = accelerator.gather_for_metrics(batch['ref'])



            # Eval mix reconstruction
            est = all_recon[:,0]
            ref = all_ref[:,0]

            am_sisdr['mix_rec'].update(- self.eval_sisdr(est=est, ref=ref).item())

            # Eval separation using mask
            all_sep_mask_norm = self.sep_norm(mix=all_ref[:,0:1], signal_sep=all_recon[:,1:])
            for i, t in enumerate(self.tracks):
                est = all_sep_mask_norm[:,i]
                ref = all_ref[:,i+1]
                ref = ref[...,:est.shape[-1]] # stft + istft. shorter

                am_sisdr[f'{t}_sep_mask'].update(- self.eval_sisdr(est=est, ref=ref).item())

            # Evaluate reconstruction of single track
            for i, t in enumerate(self.tracks):
                out_audio = self.model_eval_func((batch[t],[batch[f'{t_}/prompt'] for t_ in self.tracks]), output_tracks=[t])
                all_recon = accelerator.gather_for_metrics(out_audio)
                all_ref = accelerator.gather_for_metrics(batch[t])

                am_sisdr[f'{t}_rec'].update(- self.eval_sisdr(est=all_recon, ref=all_ref).item())

        log_dict = OrderedDict()
        for t in self.eval_tracks:

            log_dict[f'eval_{t}/sisdr'] = am_sisdr[t].avg


        log_dict['val'] = - np.mean([log_dict[f'eval_{k}/sisdr'] for k in self.eval_tracks]) # key to update best model
        
        self.scheduler_g.step(log_dict['val'])

        self.model.train()
        self.model._freeze(['encoder', 'decoder'])


        return log_dict


    @torch.no_grad()
    def run_test(self):
        self.model.eval()


        am_sisdr = {k: utils.AverageMeter() for k in self.eval_tracks}


        for batch in self.val_loader:
            
            # data transform and augmentation
            batch = self._data_transform(batch, self.cfg.training.transform, self.tm.nb_step, is_eval=True)

            # Forward
            batch = self.model(batch)

            # Distributed evaluation
            all_recon = accelerator.gather_for_metrics(batch['recon']) # valide for nested list/tuple/dict
            all_ref = accelerator.gather_for_metrics(batch['ref'])



            # Eval mix reconstruction
            est = all_recon[:,0]
            ref = all_ref[:,0]

            am_sisdr['mix_rec'].update(- self.eval_sisdr(est=est, ref=ref).item())

            



            # Eval separation using mask
            all_sep_mask_norm = self.sep_norm(mix=all_ref[:,0:1], signal_sep=all_recon[:,1:])
            for i, t in enumerate(self.tracks):
                est = all_sep_mask_norm[:,i]
                ref = all_ref[:,i+1]
                ref = ref[...,:est.shape[-1]] # stft + istft. shorter

                am_sisdr[f'{t}_sep_mask'].update(- self.eval_sisdr(est=est, ref=ref).item())

            # Evaluate reconstruction on each individual track
            for i, t in enumerate(self.tracks):
                out_audio = self.model_eval_func((batch[t],[batch[f'{t_}/prompt'] for t_ in self.tracks] ), output_tracks=[t])
                all_recon = accelerator.gather_for_metrics(out_audio)
                all_ref = accelerator.gather_for_metrics(batch[t])

                am_sisdr[f'{t}_rec'].update(- self.eval_sisdr(est=all_recon, ref=all_ref).item())

        log_dict = OrderedDict()
        for t in self.eval_tracks:

            log_dict[f'test_{t}/sisdr'] = am_sisdr[t].avg

        
        self.model.train()
        self.model._freeze(['encoder', 'decoder'])

        return log_dict
    

    @torch.no_grad()
    @accelerator.on_main_process
    def run_vis(self, nb_step):
        self.model.eval()

        ret_dict = defaultdict(list)
        
        writer = self.tracker.writer
        
        vis_idx = self.cfg.training.get('vis_idx', [])
        for idx in vis_idx:
            # get data
            batch = self.val_loader.dataset.__getitem__(idx)
            for t in self.tracks:
                batch[t] = batch[t].unsqueeze(0).to(accelerator.device) # (1, 1, T)
            # data transform and augmentation
            batch = self._data_transform(batch, self.cfg.training.transform, self.tm.nb_step, is_eval=True)
            # single track recon
            for t in self.tracks:
                ret_dict[f'{t}_orig'].append(batch[t][0])

                #print(batch['prompt'])
                ret_dict[f'{t}_recon'].append(self.model_eval_func((batch[t], [[batch[f'{t_}/prompt']] for t_ in self.tracks]), output_tracks=[t])[0])
            
            # mix recon and separation
            #print(batch['prompt'])
            sep_out = self.model_eval_func((batch['mix'],[[batch[f'{t_}/prompt']] for t_ in self.tracks]), output_tracks= ['mix'] + self.tracks)
            ret_dict['mix_orig'].append(batch['mix'][0])
            ret_dict['mix_recon'].append(sep_out[:, 0]) # (1, T)

            # separation using FFT-mask
            mix = batch['mix'].unsqueeze(2)
            signal_sep = sep_out[:,1:].unsqueeze(2)
            all_sep_mask_norm = self.sep_norm(mix, signal_sep)
            for p, t in enumerate(self.tracks):
                est = all_sep_mask_norm[0,p]
                right_pad = mix.shape[-1] - est.shape[-1]
                est = F.pad(est, (0, right_pad))
                ret_dict[f'{t}_sep_mask'].append(est)
        

        # mix
        audio_mix_orig = torch.cat(ret_dict['mix_orig'], dim=-1).detach().cpu()
        audio_mix_recon = torch.cat(ret_dict['mix_recon'], dim=-1).detach().cpu()


        audio_signal = torch.cat((audio_mix_orig, audio_mix_recon), dim=0).numpy()
        fig = utils.vis_spec(audio_signal, fs=self.sr, fig_width=8*len(vis_idx),
                             tight_layout=False, save_fig=None)
        #writer.add_figure('mix', fig, global_step=nb_step)

        # track
        for t in self.tracks:
            audio_orig = torch.cat(ret_dict[f'{t}_orig'], dim=-1).detach().cpu()
            audio_recon = torch.cat(ret_dict[f'{t}_recon'], dim=-1).detach().cpu()
            audio_sep_mask = torch.cat(ret_dict[f'{t}_sep_mask'], dim=-1).detach().cpu()

            audio_signal = torch.cat((audio_orig, audio_recon, audio_sep_mask), dim=0).numpy()
            fig = utils.vis_spec(audio_signal, fs=self.sr, fig_width=8*len(vis_idx),
                                tight_layout=False, save_fig=None)
        
        self.model.train()
        self.model._freeze(['encoder', 'decoder'])

        return
