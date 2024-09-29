import yaml
import os
import random
from tqdm import tqdm

def sort_func(json_path):
    nums = float(json_path.split('_')[-1].split('.')[0])
    return nums

if __name__ == '__main__':
    print('=== INIT DOWN ===')
    # ! set correct path
    config_path = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/config/audiosep_base_32k.yaml'
    
    # ! used datasets; datasets before "validation" are used for training
    dataset_list = ['FSD50K', 'Clotho', 'Audiocaps', 'Auto_AS_test', 'validation']  # 'Audiocaps', 'Auto_AS', 'Clotho' 
    # ! .yaml filename
    save_name = 'Fsd_Clo_Caps_Autotest_ResUNet_32k'
    
    json_list = []
    valid_json_list = []
    for dataset in dataset_list:
        print(dataset)
        # ! set correct path
        dataset_json_file = f'/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/datafiles/{dataset}'
        if 'validation' not in dataset:
            for item in tqdm(os.listdir(dataset_json_file)):
                json_list.append(f'{dataset_json_file}/{item}')
                
        else:
            for item in tqdm(os.listdir(dataset_json_file)):
                valid_json_list.append(f'{dataset_json_file}/{item}')
    
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    
    ## write file
    random.shuffle(json_list)
    data['data']['datafiles'] = json_list
    data['data']['datafiles_valid'] = valid_json_list
    data['data']['max_mix_num'] = 2         #  the max number of source audios used for mixing
    data['data']['sampling_rate'] = 32000   #  sampling rate (Hz)
    data['data']['segment_seconds'] = 10    #  input audio duration (s)
    data['data']['loudness_norm']['lower_db'] = -10  # SNR_low (dB)
    data['data']['loudness_norm']['higher_db'] = 10  # SNR_high (dB)
    

    data['train']['load_pretrain'] = True   # ! whether load pretrained model
    data['train']['batch_size_per_device'] = 3 # ! batch size
    data['train']['train_max_epoch'] = -1  # ! max train epoch: -1 means no limit
    data['train']['loss_type'] = 'l1_wav'  # ! choose your loss, see losses.py for detail
    data['train']['freeze'] = False  # whether freeze some layers in AS model, always False
    data['train']['num_workers'] = 6 # ! num_workers
    data['train']['accumulation_grad'] = 1 # ! grad accumulation times
    # ! optimizer settings, see ./optimizer/lr_schedulers.py for detail
    data['train']["optimizer"]['learning_rate'] = 1e-4 
    data['train']["optimizer"]['lr_lambda_type'] = 'constant_warm_up' 
    data['train']["optimizer"]['warm_up_steps'] = 10000
    data['train']["optimizer"]['reduce_lr_steps'] = 1000000
    

    data['model']['model_type'] = 'ResUNet30_32k'
    # ! DPRNN settings
    data['model']['dprnn'] = True
    data['model']['dprnn_layers'] = 6      # total 4 DPRNN block, each block has N*2 GRU modules
    data['model']['dprnn_hidden'] = 256    # hidden dimention in the GRU module (input dimension = 384)
    
    # ! set correct path
    config_path_new = f'/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/config/{save_name}.yaml'
    with open(config_path_new, 'w') as file:
        yaml.dump(data, file, indent=4)

    
            