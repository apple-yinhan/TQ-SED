from torch.utils.data import Dataset, DataLoader
import librosa
import config
import numpy as np
import os
import torch
from extract_features import extract_mbe, load_labels
import random

class SED(Dataset):
    def __init__(
        self,
        mel_path = '',
        soft_path = '',
        
    ):
        self.mel_list = os.listdir(mel_path)
        self.mel_path = mel_path
        self.soft_path = soft_path

    def __len__(self):
        return len(self.mel_list)

    def _read_audio(self, index):
        try:
            file_name = self.mel_list[index]
            mel = np.load(self.mel_path + f'/{file_name}')
            soft_label = np.load(self.soft_path + f'/{file_name}')

            mel = torch.from_numpy(mel)
            soft_label = torch.from_numpy(soft_label)

            hard_label = soft_label.clone()
            hard_label[hard_label>=0.5] = 1.0
            hard_label[hard_label<0.5] = 0.0
            
            asc_label = torch.ones((soft_label.shape[0],))
            if 'cafe_restaurant' in file_name:
                asc_label = asc_label * 0
            elif 'city_center' in file_name:
                asc_label = asc_label * 1
            elif 'grocery_store' in file_name:
                asc_label = asc_label * 2
            elif 'metro_station' in file_name:
                asc_label = asc_label * 3
            elif 'residential_area' in file_name:
                asc_label = asc_label * 4

            return mel.T, soft_label, hard_label, asc_label
        except Exception as e:
            print(f'Error when loading audio : {self.mel_list[index]}')
            random_indx = random.randint(0, len(self.mel_list)-1)
            return self._read_audio(index=random_indx)

    def __getitem__(self, index):
        mel, soft_label, hard_label, asc_label = self._read_audio(index)
        return mel, soft_label, hard_label, asc_label

class LASS_SED(Dataset):
    def __init__(
        self,
        mel_path = '',
        soft_path = '',
        sep_mel_path = '',
        
    ):
        self.mel_list = os.listdir(mel_path)
        self.sep_mel_path = sep_mel_path
        self.mel_path = mel_path
        self.soft_path = soft_path

    def __len__(self):
        return len(self.mel_list)

    def _read_audio(self, index):
        file_name = self.mel_list[index]
        mel = np.load(self.mel_path + f'/{file_name}')
        soft_label = np.load(self.soft_path + f'/{file_name}')

        mel = torch.from_numpy(mel)
        soft_label = torch.from_numpy(soft_label)

        hard_label = soft_label.clone()
        hard_label[hard_label>=0.5] = 1.0
        hard_label[hard_label<0.5] = 0.0

        ## load separated mels
        wavename = file_name.split('.')[0]
        idx = 0
        for event in config.labels_hard:
            sep_mel_name = wavename+f'-{event}.npy'
            sep_mel = np.load(self.sep_mel_path + f'/{sep_mel_name}')
            sep_mel = sep_mel.T
            if idx == 0:
                final_sep = sep_mel[None, :, :]
            else:
                final_sep = np.concatenate((final_sep, sep_mel[None, :, :]), axis=0)
            # print(sep_mel.shape)
            # print(final_sep.shape)
            idx += 1
        
        return mel.T, soft_label, hard_label, final_sep

    def __getitem__(self, index):
        mel, soft_label, hard_label, final_sep = self._read_audio(index)
        return mel, soft_label, hard_label, final_sep


if __name__ == '__main__':
    dataset = SED('/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline/fold1/train/mel', '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline/fold1/train/soft')
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    for mel, soft_label, hard_label, asc_label in dataloader:
        print(mel.shape, soft_label.shape, hard_label.shape)
        print(soft_label[0,0,:], hard_label[0,0,:])
        break
    dataset = LASS_SED('/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline/fold1/train/mel', 
                       '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline/fold1/train/soft',
                       '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline/fold1/train/sep_mel_32k'
                      )
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    for mel, soft_label, hard_label, final_sep in dataloader:
        print(mel.shape, soft_label.shape, hard_label.shape, final_sep.shape)
        print(soft_label[0,0,:], hard_label[0,0,:])
        break