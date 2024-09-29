import os
import sys
import re
import json
from typing import Dict, List
import soundfile as sf
import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import librosa
import torchaudio
#import lightning.pytorch as pl
import pytorch_lightning as pl
from models.clap_encoder import CLAP_Encoder

# sys.path.append('../dcase2024_task9_baseline/')
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)



class DCASEEvaluator:
    def __init__(
        self,
        model_input_fs = 16000,
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
        demo_dir='demos',
        hybrid=False,
        demo_number = 15
    ) -> None:
        r"""DCASE T9 LASS evaluator.

        Returns:
            None
        """
        self.model_input_fs = model_input_fs
        self.sampling_rate = sampling_rate
        self.demo_num = demo_number
        with open(eval_indexes) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        
        self.eval_list = eval_list
        self.audio_dir = audio_dir
        self.demo_dir = demo_dir
        self.hybrid = hybrid
        os.makedirs(demo_dir, exist_ok=True)

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        print(f'Evaluation on DCASE T9 synthetic validation set.')
        
        pl_model.eval()
        device = pl_model.device

        sisdrs_list = []
        sdris_list = []
        sdrs_list = []
        
        demo_num = 1
        demo_max = self.demo_num  # 最大demo数
        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                source, noise, snr, caption = eval_data
                snr = int(snr)

                source_path = os.path.join(self.audio_dir, f'{source}.wav')
                noise_path = os.path.join(self.audio_dir, f'{noise}.wav')

                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                noise, fs = librosa.load(noise_path, sr=self.sampling_rate, mono=True)

                # create audio mixture with a specific SNR level
                source_power = np.mean(source ** 2)
                noise_power = np.mean(noise ** 2)
                desired_noise_power = source_power / (10 ** (snr / 10))
                scaling_factor = np.sqrt(desired_noise_power / noise_power)
                noise = noise * scaling_factor

                mixture = source + noise

                # declipping if need be
                max_value = np.max(np.abs(mixture))
                if max_value > 1:
                    source *= 0.9 / max_value
                    mixture *= 0.9 / max_value

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                conditions = pl_model.query_encoder.get_query_embed(
                    modality='text',
                    text=[caption],
                    device=device 
                )
                
            
                if self.sampling_rate != self.model_input_fs:
                    # ---------resample input waveform ------------
                    input_wave = torch.Tensor(mixture).to(device)
                    input_wave = torchaudio.functional.resample(input_wave, orig_freq=self.sampling_rate, 
                                                                new_freq=self.model_input_fs)
                    # print(input_wave.shape)
                    input_wave = input_wave[None, None, :]
                else:
                    input_wave = torch.Tensor(mixture)[None, None, :].to(device)
                # ---------------------------------------------
                input_dict = {
                    "mixture": input_wave,
                    "condition": conditions,
                }
                
                outputs = pl_model.ss_model(input_dict)

                if self.sampling_rate != self.model_input_fs:
                    # resample output
                    sep_segment = outputs["waveform"].squeeze(0).squeeze(0)
                    sep_segment = torchaudio.functional.resample(sep_segment, orig_freq=self.model_input_fs, 
                                                                new_freq=self.sampling_rate)
                    sep_segment = sep_segment[None, None, :]
                else:
                    sep_segment = outputs["waveform"]

                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                # sep_segment: (segment_samples,)
                
                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)

                sisdrs_list.append(sisdr)
                sdris_list.append(sdri)
                sdrs_list.append(sdr)
                
                # output demos
                if demo_num <= demo_max:
                    sf.write(f'{self.demo_dir}/mixture_{demo_num}.wav', 
                             mixture, fs , 'PCM_16')
                    sf.write(f'{self.demo_dir}/output_{demo_num}.wav', 
                             sep_segment, fs , 'PCM_16')
                    if self.hybrid:
                        spec_out = outputs['spec_out']
                        spec_out = spec_out.squeeze(0).squeeze(0).data.cpu().numpy()
                        
                        wave_out = outputs['wave_out']
                        wave_out = wave_out.squeeze(0).squeeze(0).data.cpu().numpy()
                        
                        sf.write(f'{self.demo_dir}/specout_{demo_num}.wav', 
                                 spec_out, fs , 'PCM_16')
                        sf.write(f'{self.demo_dir}/waveout_{demo_num}.wav', 
                                 wave_out, fs , 'PCM_16')
                    with open(f'{self.demo_dir}/caption_{demo_num}.txt', 'w') as f:
                        f.write(caption+f'|noise-SNR:{snr}|SDR:{sdr}')
                    demo_num += 1
                    
        mean_sdri = np.mean(sdris_list)
        mean_sisdr = np.mean(sisdrs_list)
        mean_sdr = np.mean(sdrs_list)
        
        return mean_sisdr, mean_sdri, mean_sdr
    


def eval(evaluator, checkpoint_path, config_yaml='./config/audiosep_base.yaml', device = "cuda"):
    configs = parse_yaml(config_yaml)

    # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'-------  Start Evaluation  -------')

    # evaluation 
    SISDR, SDRi, SDR = evaluator(pl_model)
    msg_clotho = "SDR: {:.3f}, SDRi: {:.3f}, SISDR: {:.3f}".format(SDR, SDRi, SISDR)
    print(msg_clotho)

    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    print('=====LOAD DOWN=====')
    system = 'baseline'
    model_name = 'audiosep_base_4M_steps.ckpt'
    config_name = 'Fsd_Clo_Caps_Autotest_ResUNet_32k_nodprnn.yaml'
    
    checkpoint_path=f'workspace/checkpoints/{system}/{model_name}'
    config_yaml=f'config/{config_name}'
    
    demo_numbers = 15

    hybrid_out = False
    dcase_evaluator = DCASEEvaluator(
        model_input_fs = 32000,
        sampling_rate=16000,
        eval_indexes='./dataset/validation/lass_synthetic_validation.csv',
        audio_dir='./dataset/validation/lass_validation',
        demo_dir=f'./demo/{system}',
        hybrid=(hybrid_out),
        demo_number=demo_numbers
    )
    eval(dcase_evaluator, checkpoint_path, config_yaml, device = "cuda:0")
    