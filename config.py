exp_id = 27
folder_idx = 'fold1'
backbone = 'crnn' # idc_sdme
batchsize = 16
device = 'cuda:0'
posterior_thresh = 0.5
lass_sr = 32000
lass_duration = 10
audio_sr = 44100
hop_len = 8820 #11025
segment = int(39.9 * audio_sr) # 24.9
segment_hop = int(1 * audio_sr)
nfft = int(hop_len * 2)
nb_mel_bands = 64#128 
is_mono = True
fmin = 50 #10
fmax = 14000 #20000
class_nums = 11
cnn_filters=128
rnn_hid=32#64
_dropout_rate=0.1#0.1
finetune = False
patience = 10
lr = 1e-3
load_model = False
mixup = False
lass_sed_type = 'Method_A'  # Method_A Method_B Method_C
path_groundtruth = '/mnt/nfs2/hanyin/LASS4SED/baseline-codes/task4b_mycodes/metadata/gt_dev.csv'
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
