import json
import os
import sys
import pandas as pd
import torchaudio
from tqdm import tqdm
if __name__ == '__main__':
    print('=== INIT DOWN ===')
    dataset = ['FSD50K']   # Clotho or FSD50K or validation
    save_filename = dataset[0] 
    output_file = f'/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/datafiles/{save_filename}'
    os.makedirs(output_file, exist_ok = True)

    dataset_file = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/dataset/development'
    temp_json_path = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/datafiles/template.json'

    with open(temp_json_path, 'r') as file:
        temp_data = json.load(file)

    for data_source in dataset:
        print(f'=== Processing {data_source} ===')
        audio_idx = 1
        if data_source == 'Clotho':
            caption_per_audio = 5
            filenames = ['development', 'evaluation', 'validation']  # 'development', 'evaluation', 
            for filename in filenames:
                print(f'=== Processing {filename} ===')
                audio_file = f'{dataset_file}/{data_source}/clotho_audio_{filename}/{filename}'
                caption_table = pd.read_csv(f'{dataset_file}/{data_source}/clotho_captions_{filename}.csv')  # N * 6
                for audioname in tqdm(os.listdir(audio_file)):
                    audio_path = f'{audio_file}/{audioname}'
                    audio_wave, fs = torchaudio.load(audio_path, channels_first=True)
                    if audio_wave.shape[1] > fs * 0.5: # duration<1s drop ...
                        for caption_num in range(caption_per_audio):
                            audio_caption = caption_table[caption_table['file_name']==audioname].iloc[0,caption_num+1]
                            
                            json_data = temp_data
                            json_data['data'][0]['wav'] = audio_path
                            json_data['data'][0]['caption'] = audio_caption
                            
                            output_file_path = output_file + f'/audio_{audio_idx}.json'
                            with open(output_file_path, 'w') as file:
                                json.dump(json_data, file, indent=2)
                            
                            audio_idx += 1
        elif data_source == 'FSD50K':
            filenames = ['evaluation', 'development']
            for filename in filenames:
                print(f'=== Processing {filename} ===')
                if filename == 'development':
                    audio_file = f'{dataset_file}/{data_source}/FSD50K-{filename}/FSD50K.dev_audio/FSD50K.dev_audio'
                    caption_path = f'{dataset_file}/{data_source}/FSD50K.ground_truth/fsd50k_dev_auto_caption.json'
                else:
                    audio_file = f'{dataset_file}/{data_source}/FSD50K-{filename}/FSD50K.eval_audio/FSD50K.eval_audio'
                    caption_path = f'{dataset_file}/{data_source}/FSD50K.ground_truth/fsd50k_eval_auto_caption.json'
                with open(caption_path, 'r') as file:
                    caption_data = json.load(file)
                caption_data = caption_data['data']
                wavename_list = []
                caption_list = []
                for item in caption_data:
                    wavename_list.append(item['wav'])
                    caption_list.append(item['caption'])
                    
                d = {'audioname': wavename_list, 'caption': caption_list}
                caption_table = pd.DataFrame(data=d)
                
                for audioname in tqdm(os.listdir(audio_file)):
                    audio_path = f'{audio_file}/{audioname}'
                    audio_wave, fs =  torchaudio.load(audio_path, channels_first=True)
                    if audio_wave.shape[1] > fs * 0.5: # duration<0.5s drop ...
                        audio_caption = caption_table[caption_table['audioname']==audioname].iloc[0,1]
                        
                        json_data = temp_data
                        json_data['data'][0]['wav'] = audio_path
                        json_data['data'][0]['caption'] = audio_caption
                        
                        output_file_path = output_file + f'/audio_{audio_idx}.json'
                        with open(output_file_path, 'w') as file:
                            json.dump(json_data, file, indent=2)
                        
                        audio_idx += 1
                        # sys.exit()
                    
        elif data_source == 'validation':
            dataset_file = dataset_file.replace('development', 'validation')
            audio_file = f'{dataset_file}/lass_validation'
            caption_path = f'{dataset_file}/lass_synthetic_validation.csv'
            caption_table = pd.read_csv(caption_path) 
            caption_per_audio = 3
            for audioname in os.listdir(audio_file):
                audio_path = f'{audio_file}/{audioname}'
                for caption_num in range(caption_per_audio):
                    
                    # print(audioname)
                    audioname_new = audioname[:-4]
                    audio_caption = caption_table[caption_table['source']==audioname_new]
                    audio_caption = audio_caption.iloc[caption_num, -1]
                    
                    json_data = temp_data
                    json_data['data'][0]['wav'] = audio_path
                    json_data['data'][0]['caption'] = audio_caption
                    
                    output_file_path = output_file + f'/audio_{audio_idx}.json'
                    with open(output_file_path, 'w') as file:
                        json.dump(json_data, file, indent=2)
                    
                    audio_idx += 1


   