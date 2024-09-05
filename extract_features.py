import numpy as np
import librosa
import config


def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)
    mbe = np.dot(mel_basis, spec)
    mbe = librosa.amplitude_to_db(mbe, ref=np.max)*(-1)
    return mbe

def load_labels(file_name, nframes, soft_flag=True):
    annotations = []
    for l in open(file_name):
        words = l.strip().split('\t')
        if soft_flag:
            annotations.append([float(words[0]), float(words[1]), 
                                config.class_labels_soft[words[2]], float(words[3])])
        else:
            annotations.append([float(words[0]), float(words[1]), 
                                config.class_labels_hard[words[2]], float(words[3])])
    # Initialize label matrix
    if soft_flag:
        label = np.zeros((nframes, len(config.class_labels_soft)))
    else:
        label = np.zeros((nframes, len(config.class_labels_hard)))
    tmp_data = np.array(annotations)
    
    frame_start = np.floor(tmp_data[:, 0] * config.audio_sr / config.hop_len).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * config.audio_sr / config.hop_len).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = tmp_data[:, 3][ind]

    return label