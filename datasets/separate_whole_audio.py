import os
import sys
sys.path.append('/mnt/nfs2/hanyin/LASS4SED')
import config
import torchaudio
from LASS_codes.models.clap_encoder import CLAP_Encoder
import soundfile as sf
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)
import torch

audio_folder_1 = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_audio'
separated_audio_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_audio/separated_audio_16k'
lass_sr = config.lass_sr
lass_dur = config.lass_duration
print('LASS sampling rate:', lass_sr)
device = 'cuda:0'
if lass_sr == 32000:
    config_yaml = '/mnt/nfs2/hanyin/LASS4SED/LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_32k.yaml'
    checkpoint_path = '/mnt/nfs2/hanyin/LASS4SED/LASS_codes/checkpoints/resunet_with_dprnn_32k/model-epoch=01-val_sdr=8.6049.ckpt'
elif lass_sr == 16000:
    config_yaml = '/mnt/nfs2/hanyin/LASS4SED/LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_16k.yaml'
    checkpoint_path = '/mnt/nfs2/hanyin/LASS4SED/LASS_codes/checkpoints/resunet_with_dprnn_16k/model-epoch=19-val_sdr=8.1018.ckpt'
configs = parse_yaml(config_yaml)
# Load model
query_encoder = CLAP_Encoder().eval()

pl_model = load_ss_model(
    configs=configs,
    checkpoint_path=checkpoint_path,
    query_encoder=query_encoder
).to(device)

pl_model.eval()
with torch.no_grad():
    for place in os.listdir(audio_folder_1):
        if 'separated' not in place:
            print(place)
            audio_folder_2 = audio_folder_1 + f'/{place}'
            for wavename in os.listdir(audio_folder_2):
                audio, sr = torchaudio.load(audio_folder_2 + f'/{wavename}', channels_first=True)
                if audio.shape[0] == 2:
                    audio = (audio[0,:]+audio[1,:])/2
                audio = audio.reshape(1,-1) # [1, samples]

                if sr != lass_sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=lass_sr)

                for caption in config.labels_hard:
                    conditions = pl_model.query_encoder.get_query_embed(
                                        modality='text',
                                        text=[caption],
                                        device=device 
                                    )

                    n_sample = lass_sr * lass_dur
                    nums = audio.shape[-1] // n_sample
                    sep_savename = f'{wavename[:-4]}-{caption}'
                    for i in range(nums):
                        segment = audio[:, i*n_sample:(i+1)*n_sample]
                        segment = segment.to(device)  # print(segment.shape) [1, 320000]
                        input_dict = {
                                        "mixture": segment[None, :, :],
                                        "condition": conditions,
                                    }
                        
                        outputs = pl_model.ss_model(input_dict)
                        sep_segment = outputs["waveform"]  # print(sep_segment.shape) [1, 1, 320000]
                        sep_segment = sep_segment.squeeze(0)
                        ## concatenate
                        if i == 0:
                            final_segment = sep_segment
                        else:
                            final_segment = torch.cat((final_segment, sep_segment), dim=-1)
                    if (audio.shape[-1] - (i+1)*n_sample) > 0:
                        segment = audio[:, (i+1)*n_sample: ]
                        segment = segment.to(device)  # print(segment.shape) [1, n]
                        rest_sample = segment.shape[-1]

                        segment_pad = torch.zeros((1, lass_sr * lass_dur)).to(device)
                        segment_pad[:, :rest_sample] = segment
                        input_dict = {
                                        "mixture": segment_pad[None, :, :],
                                        "condition": conditions,
                                    }
                        
                        outputs = pl_model.ss_model(input_dict)
                        sep_segment = outputs["waveform"]  # print(sep_segment.shape) [1, 1, 320000]
                        sep_segment = sep_segment.squeeze(0)
                        sep_segment = sep_segment[:, :rest_sample]
                        final_segment = torch.cat((final_segment, sep_segment), dim=-1)
                    
                    # print(final_segment.shape)
                    # print(audio.shape)

                    if sr != lass_sr:
                        final_segment = torchaudio.functional.resample(final_segment, orig_freq=lass_sr, new_freq=sr)

                    final_segment = final_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                    sf.write(f'{separated_audio_folder}/{sep_savename}.wav', final_segment, sr , 'PCM_32')
                    # sys.exit()

                

