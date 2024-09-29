exp_id = 31
folder_idx = 'fold5'
backbone = 'crnn' 
batchsize = 16
device = 'cuda:1'
posterior_thresh = 0.5
lass_sr = 32000
lass_duration = 10
audio_sr = 44100
hop_len = 8820 
segment = int(39.9 * audio_sr) 
segment_hop = int(1 * audio_sr)
nfft = int(hop_len * 2)
nb_mel_bands = 64
is_mono = True
fmin = 50 
fmax = 14000 
class_nums = 11
cnn_filters=128
rnn_hid=32
_dropout_rate=0.2
finetune = False
patience = 10
lr = 1e-3
load_model = False
lass_sed_type = 'Method_A'  
dataset_audio_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_audio'
dataset_label_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_annotation'
dataset_folds_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_folds'
preprocessed_data_folder = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_split_baseline'
path_groundtruth = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/metadata/gt_dev.csv'
# ===================================
labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 
               'children voices', 'wind_blowing', 'brakes_squeaking', 
               'large_vehicle', 'cutlery and dishes', 'metro approaching', 
               'metro leaving']
class_labels_hard = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'metro approaching': 9,
    'metro leaving': 10,
    }
labels_soft = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 
               'wind_blowing', 'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 
               'metro approaching', 'metro leaving', 'furniture dragging', 'coffee machine',
               'door opens/closes', 'announcement', 'shopping cart', 'cash register beeping']
class_labels_soft = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'metro approaching': 9,
    'metro leaving': 10,
    'furniture dragging': 11,
    'coffee machine': 12,
    'door opens/closes': 13,
    'announcement': 14,
    'shopping cart': 15,
    'cash register beeping': 16
}
