# Text-Queried Sound Event Detection
This is the official code for paper **"Exploring Text-Queried Sound Event Detection with Audio Source Separation"**.

Paper Link: [https://arxiv.org/abs/2409.13292](https://arxiv.org/abs/2409.13292)

- Please first `pip install -r requirements.txt`

## 1. Pre-train LASS model
Please follow instructions in DCASE 2024 Challenge Task 9 Baseline for pre-training LASS models: [https://github.com/Audio-AGI/dcase2024_task9_baseline](https://github.com/Audio-AGI/dcase2024_task9_baseline)
                                                                                                  
We only made **2** alterations over the baseline:

**Baseline Model: AudioSep -> AudioSep-DP**

Codes for 16 kHz and 32 kHz AudioSep-DP are in `./LASS_codes/models/resunet.py` and `./LASS_codes/models/resunet_32k.py`.

**Training Data: FSD50K, Clotho -> FSD50K, Clotho, AudioCaps, Auto-ACD, WavCaps**
  
Link to AudioCaps: [https://audiocaps.github.io/](https://audiocaps.github.io/)

Link to Auto-ACD: [https://auto-acd.github.io/](https://auto-acd.github.io/)
                                                                                                  
Link to WaveCaps: [https://github.com/XinhaoMei/WavCaps](https://github.com/XinhaoMei/WavCaps)                                                                                         
                                                                                                  
**Or you can directly use our pre-trained models**, we release our pre-trained AudioSep-DP models (16 kHz and 32 kHz) at: [https://zenodo.org/records/14208090](https://zenodo.org/records/14208090)                                                                                         

## 2. Text-Queried Sound Event Detection

### 2.1 Prepare data 

#### 2.1.1 Download SED dataset                                                                       
                                                                       
In our paper, we used **MAESTRO Real** dataset for TQ-SED. Please download dataset from [https://zenodo.org/records/7244360](https://zenodo.org/records/7244360), and put the data under *./datasets/maestro_real/*   

Your dataset folder should be like:
                                                                       
- ./datasets/maestro_real/metadata/{development_metadata.csv, gt_dev.csv}

- ./datasets/maestro_real/development_annotation/

- ./datasets/maestro_real/development_audio/
  
- ./datasets/maestro_real/development_folds/
                                              
#### 2.1.2 Split data into 5 folds

In this study, we used a 5 folds cross-validation set up. So please use `./datasets/process_datas.py` to split your data into 5 folds according to the metadata. 
Also, this script will split long audio into short clips.

              
#### 2.1.3 Extract mel features

Pleasr use `./datasets/extract_feat.py` for extracting the mel features.

#### 2.1.4 Use pre-trained LASS models for separation

In our work, we used the pre-trained LASS model to extract sound tracks of different events. Please use `./datasets/separate_audio.py` and `./datasets/separate_whole_audio.py` 
to perform this procedure.

PS: For separate_audio.py, we separate the short audio clips under each folder (fold1, fold2, fold3, fold4, fold5). For separate_whole_audio.py, we separate those long clips in the original dataset folder. Because during evaluation, we input the whole clip into the SED model, not those splited short clips.
                                                                                                  
### 2.2 Training

**Conventional SED Framework**: run `python train_sed.py`

**Proposed TQ-SED Framework**: run `python train_tq_sed.py`
                                                                                                  
### 2.3 Testing                                                                                                
                                                                                                  
**Conventional SED Framework**: run `python test_sed.py`

**Proposed TQ-SED Framework**: run `python test_tq_sed.py`         

- PS: For TQ-SED, we have tried 4 structures, referred to `Mthoda A,B,C,D` in the code. The proposed method is `Method A`.

## 3. Appreciation

This project is built on the following projects, we appreciate their hard-working:

- [https://github.com/Audio-AGI/dcase2024_task9_baseline](https://github.com/Audio-AGI/dcase2024_task9_baseline)
   
- [https://audiocaps.github.io/](https://audiocaps.github.io/)
   
- [https://auto-acd.github.io/](https://auto-acd.github.io/)
   
- [https://github.com/XinhaoMei/WavCaps](https://github.com/XinhaoMei/WavCaps)
   
- [https://zenodo.org/records/7244360](https://zenodo.org/records/7244360)
   
- [https://github.com/marmoi/dcase2023_task4b_baseline](https://github.com/marmoi/dcase2023_task4b_baseline)
   
- [https://github.com/yluo42/TAC](https://github.com/yluo42/TAC)
   
- [https://github.com/fgnt/sed_scores_eval](https://github.com/fgnt/sed_scores_eval)

