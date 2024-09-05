from LASS_codes.models.clap_encoder import CLAP_Encoder
import torchaudio
import soundfile as sf
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)
import torch


if __name__ == '__main__':
    print('=== separate begin ===')
    device = 'cuda:0'
    model_sr = 32000
    duration = 10

    config_yaml = './LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_32k.yaml'
    checkpoint_path = './LASS_codes/checkpoints/resunet_with_dprnn_32k/model-epoch=01-val_sdr=8.6049.ckpt'

    # config_yaml = './LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_32k_nodprnn.yaml'
    # config_yaml = './LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_16k.yaml'
    # config_yaml = './LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_16k_nodprnn.yaml'
    
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
        # load audio and captions
        audio_path = './separate_demo/16k.wav'
        audio, sr = torchaudio.load(audio_path, channels_first=True)

        if audio.shape[0] == 2:
            audio = (audio[0,:]+audio[1,:])/2
        audio = audio.reshape(1,-1) # [1, samples]

        if sr != model_sr:
            audio_2 = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=model_sr)
        else:
            audio_2 = audio
        
        audio_2 = audio_2[:, (0*model_sr):((0+duration)*model_sr)]
        audio_2 = audio_2.to(device)
        caption = 'bird singing'
        conditions = pl_model.query_encoder.get_query_embed(
                        modality='text',
                        text=[caption],
                        device=device 
                    )

        input_dict = {
                        "mixture": audio_2[None, :, :],
                        "condition": conditions,
                    }
        
        outputs = pl_model.ss_model(input_dict)
        sep_segment = outputs["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
        sf.write('./separate_demo/sep_bird.wav', sep_segment, model_sr , 'PCM_32')




