import librosa
import numpy as np
import wave
import os
import sys
sys.path.append('/mnt/nfs2/hanyin/LASS4SED')
import config as args
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from extract_features import extract_mbe

if __name__ == '__main__':
    # audio_path = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_audio/cafe_restaurant/cafe_restaurant_00.wav'
    # wave, _ = librosa.load(audio_path, sr=args.audio_sr, mono=True)
    # wave = wave[:args.segment]
    # print(wave.shape)
    # mel = extract_mbe(wave, args.audio_sr, args.nfft, args.hop_len, args.nb_mel_bands, args.fmin, args.fmax)
    # print(mel.shape) # [nmel, frames]

    print(' === process train/val audio for each fold ===')
    split_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline'
    os.makedirs(split_folder, exist_ok=True)
    for fold_idx in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
        for type_ in ['train', 'val']:
            print(f'=== {fold_idx} - {type_} ===')
            total_idx = 0
            saved_audio_folder = f'{split_folder}/{fold_idx}/{type_}/audio'
            saved_label_folder = f'{split_folder}/{fold_idx}/{type_}/label'
            os.makedirs(saved_audio_folder, exist_ok=True)
            os.makedirs(saved_label_folder, exist_ok=True)

            meta_path = f'/mnt/nfs2/hanyin/LASS4SED/baseline-codes/task4b_official/development_folds/{fold_idx}_{type_}.csv'
            meta_data = pd.read_csv(meta_path)
            for info in tqdm(meta_data['filename']):
                place = info.split('/')[0]
                wavename = info.split('/')[1]
                audio_path = f'/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_audio/{place}/{wavename}.wav'
                wave, _ = librosa.load(audio_path, sr=args.audio_sr, mono=True)
                # print(wave.shape) # [samples, ]
                label_path = f'/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_annotation/soft_labels_{place}/{wavename}.txt'
                label = pd.read_csv(label_path, delimiter='\t', header=None)
                # print(label)  # [nums * 4] onset, offset. event, prob

                ## splt audio and meta
                split_num = (wave.shape[0] - args.segment) // args.segment_hop + 1
                # print(wave.shape, split_num, args.segment, split_num * args.segment)
                
                for i in range(split_num):
                    onset_idx = i*args.segment_hop
                    offset_idx = i*args.segment_hop + args.segment
                    ## write split audio
                    wave_split = wave[onset_idx:offset_idx]
                    wave_split = wave_split.reshape((-1, 1))
                    sf.write(f'{saved_audio_folder}/{total_idx}|{wavename}.wav', wave_split, args.audio_sr, 'PCM_32')
                    
                    ## extract labels and write
                    onset_time = onset_idx / args.audio_sr
                    offset_time = offset_idx / args.audio_sr

                    finded_onsets = []
                    finded_offsets = []
                    finded_events = []
                    finded_probs = []
                    for j in range(label.shape[0]):
                        onset_, offset_, event_, prob_ = label.iloc[j, 0], label.iloc[j, 1], label.iloc[j, 2], label.iloc[j, 3]
                        if onset_ >= onset_time and offset_ <= offset_time:
                            finded_onsets.append(onset_ - onset_time)
                            finded_offsets.append(offset_ - onset_time) 
                            finded_events.append(event_)
                            finded_probs.append(prob_)
                    
                    with open(f'{saved_label_folder}/{total_idx}|{wavename}.txt', 'w') as f:
                        for m in range(len(finded_events)):
                            f.write(str(finded_onsets[m]))
                            f.write('\t')
                            f.write(str(finded_offsets[m]))
                            f.write('\t')
                            f.write(finded_events[m])
                            f.write('\t')
                            f.write(str(finded_probs[m]))
                            f.write('\n')
                            
                    total_idx += 1
                    # print(onset_time, offset_time)
    # sys.exit()

                    

                
            


