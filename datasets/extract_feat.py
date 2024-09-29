import os
import sys
sys.path.append('/mnt/nfs2/hanyin/LASS4SED')
from extract_features import extract_mbe, load_labels
from tqdm import tqdm
import librosa
import config
import sys
import numpy as np

split_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_mine'
os.makedirs(split_folder, exist_ok=True)
for fold_idx in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
    print(f'==={fold_idx}===')
    data_folder1 = split_folder + f'/{fold_idx}'
    for type_ in ['train', 'val']:
        print(f'==={type_}===')
        data_folder2 = data_folder1 + f'/{type_}'
        audio_folder = data_folder2 + '/audio'
        label_folder = data_folder2 + '/label'
        saved_mel_folder = data_folder2 + '/mel'
        saved_soft_folder = data_folder2 + '/soft'
        os.makedirs(saved_mel_folder, exist_ok=True)
        os.makedirs(saved_soft_folder, exist_ok=True)
        for wavename in tqdm(os.listdir(audio_folder)):
            labelname = wavename.replace('wav', 'txt')
            wave, _ = librosa.load(f'{audio_folder}/{wavename}', sr=config.audio_sr, mono=True)
            mel = extract_mbe(wave, config.audio_sr, config.nfft, config.hop_len, config.nb_mel_bands, config.fmin, config.fmax) # [nmel, nframes]
            nframes = mel.shape[1]

            soft_label = load_labels(f'{label_folder}/{labelname}', nframes, True)

            # print(mel.shape) [64, 200]
            # print(soft_label.shape) [200, 17]
            npy_name = wavename.replace('wav', 'npy')
            np.save(saved_mel_folder + f'/{npy_name}', mel)
            np.save(saved_soft_folder + f'/{npy_name}', soft_label)
            # print(mel.shape, soft_label.shape)
            # sys.exit()