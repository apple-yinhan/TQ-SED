# from typing import Any, Callable, Dict
import random
# import lightning.pytorch as pl
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from .clap_encoder import CLAP_Encoder

from huggingface_hub import PyTorchModelHubMixin

# hanyin copy from utils.py
def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference


    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)

    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

    sdr = 10. * np.log10(numerator / denominator)

    return sdr
# ------------------------

class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        waveform_mixer = None,
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float =1.0,
        freeze = False,
        batchsize = 6, 
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        self.freeze = freeze
        self.batchsize = batchsize

    def forward(self, x):
        pass
    
    def validation_step(self, batch_data_dict, batch_idx):
        # Created by Han Yin, 04/12/2024
        r"""Forward a mini-batch data to model, calculate loss function, and
        validate for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. 
                'audio_text': {
                    'text': ['a sound of dog', ...]
                    'waveform': (batch_size, 1, samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_audio_text_dict = batch_data_dict['audio_text']

        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        device = batch_audio.device
        
        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio,
            valid=True
        )

        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='hybird',
                text=batch_text,
                audio=segments.squeeze(1),
                use_text_ratio=self.use_text_ratio,
            )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.eval()
        with torch.no_grad():
            sep_segment = self.ss_model(input_dict)['waveform']
        sep_segment = sep_segment.squeeze() # (batch_size, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        # loss = self.loss_function(output_dict, target_dict)
        # self.log_dict({"val_loss": loss}, batch_size=(self.batchsize))
        
        # Caculate SDR.
        output = output_dict['segment']
        target = target_dict['segment']

        sdr = 0
        for i in range(output.shape[0]):
            spe_wav = output[i].reshape(-1,).data.cpu().numpy()
            ref_wav = target[i].reshape(-1,).data.cpu().numpy()
            sdr += calculate_sdr(ref=ref_wav, est=spe_wav)
        sdr = sdr / output.shape[0]
        sdr = torch.tensor(sdr)
        self.log_dict({"val_sdr": sdr}, batch_size=(self.batchsize))
        return sdr
    
    def validation_epoch_end(self, outputs):
        val_sdr_mean = torch.stack(outputs).mean()
        print('Val sdr mean', val_sdr_mean)
        self.log('val_sdr', val_sdr_mean, batch_size=(self.batchsize))
 
    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. 
                'audio_text': {
                    'text': ['a sound of dog', ...]
                    'waveform': (batch_size, 1, samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_audio_text_dict = batch_data_dict['audio_text']

        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        device = batch_audio.device
        
        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio,
            valid=False
        )

        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='hybird',
                text=batch_text,
                audio=segments.squeeze(1),
                use_text_ratio=self.use_text_ratio,
            )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform'] # (batch_size, 1, segment_samples)
        sep_segment = sep_segment.squeeze() # (batch_size, segment_samples)
        
        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)

        self.log_dict({"train_loss": loss}, batch_size=(self.batchsize))
        
        return loss

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.freeze:  # freeze ss_model layers
            print('=================== FREEZE Some Modules !!!!!! ===================')
            for param in self.ss_model.named_parameters(): 
                if 'DPRNN' not in param[0]:
                    param[1].requires_grad = False
    
            optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, 
                                           self.ss_model.parameters()), 
                           lr=self.learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0.0,
                           amsgrad=True,)
        else:
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )


        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    

def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from .resunet import ResUNet30
        return ResUNet30
    elif model_type == 'ResUNet30_32k':
        from .resunet_32k import ResUNet30
        return ResUNet30
    elif model_type == 'HResUNet30':
        from .resunet import HResUNet30
        return HResUNet30
    
    else:
        raise NotImplementedError
