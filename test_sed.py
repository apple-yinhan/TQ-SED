from evaluate import eval_meta, process_event, get_PSDS
import config
import numpy as np
import sed_eval
import os
import torch
from extract_features import extract_mbe, load_labels
import librosa
from tqdm import tqdm
import sys
import pandas as pd
import copy

def apply_model(model, inputs, asc_label=None, seq_len=200, device='cpu'):
    inputs = torch.from_numpy(inputs[None,:,:]).float().to(device)
    if asc_label is not None:
        asc_label = asc_label[None, :].long().to(device)
    model.eval()
    with torch.no_grad():
        frames = inputs.shape[1]
        num = frames // seq_len
        
        more = frames % seq_len
        for i in range(num):
            input_data = inputs[:, i*seq_len:(i+1)*seq_len, :]
            if asc_label is not None:
                asc_data = asc_label[:, i*seq_len:(i+1)*seq_len]
                out_data = model(input_data, asc_data, True)
                out_data = out_data[2]
            else:
                out_data = model(input_data)
                out_data = out_data
            if i == 0:
                final_out = out_data
            else:
                final_out = torch.cat((final_out, out_data), dim=1)
        if more != 0:
            input_data = inputs[:, -1 - more:-1, :]
            if asc_label is not None:
                asc_data = asc_label[:, -1 - more:-1]
                out_data = model(input_data, asc_data, True)
                out_data = out_data[2]
            else:
                out_data = model(input_data)
                out_data = out_data
            final_out = torch.cat((final_out, out_data), dim=1)

    return final_out

if __name__ == '__main__':
    print('==============================================================')
    asc_mask = False # ignore
    nbatch = 0
    segment_based_metrics_test = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=config.labels_hard,
        time_resolution=1.0
    )
    for folders in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:#
        nbatch_ = 0

        segment_based_metrics_fold = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=config.labels_hard,
            time_resolution=1.0
        )
        
        device = config.device
        # Load best model based on validation loss
        if config.backbone == 'crnn':
            from models.sed import CRNN
            model = CRNN(classes_num=config.class_nums, cnn_filters=config.cnn_filters, rnn_hid=config.rnn_hid, _dropout_rate=config._dropout_rate)

        # test audio information
        test_meta_path = f'{config.dataset_folds_folder}/{folders}_test.csv'
        test_meta = pd.read_csv(test_meta_path)
        
        out_folder = './experiments/exp{}/{}'.format(config.exp_id, folders)
        os.makedirs(out_folder, exist_ok=True)
        #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
        print("===== TESTING FOLD :{} =====".format(folders))
        model_path = out_folder + '/{}_sed.pth'.format(config.backbone)
        print("Continuing training full model from checkpoint " + model_path)
        model.load_state_dict(torch.load(model_path))

        result_folder = './experiments/exp{}/outputs'.format(config.exp_id)
        os.makedirs(result_folder, exist_ok=True)

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for item in tqdm(test_meta['filename']):
                place = item.split('/')[0]
                audioname = item.split('/')[1]

                audio_path = f'{config.dataset_audio_folder}/{place}/{audioname}.wav'
                label_path = f'{config.dataset_label_folder}/soft_labels_{place}/{audioname}.txt'

                wave, _ = librosa.load(audio_path, sr=config.audio_sr, mono=True)
                feat = extract_mbe(wave, config.audio_sr, config.nfft, config.hop_len, config.nb_mel_bands, config.fmin, config.fmax)
                nframes = feat.shape[1]
                soft_label = load_labels(label_path, nframes, True)
                hard_label = copy.deepcopy(soft_label)
                hard_label[hard_label >= 0.5] = 1.0
                hard_label[hard_label < 0.5] = 0.0
                hard_label = hard_label[:,:11]

                if asc_mask:
                    asc_label = torch.ones((soft_label.shape[0],))
                    if 'cafe_restaurant' in audioname:
                        asc_label = asc_label * 0
                    elif 'city_center' in audioname:
                        asc_label = asc_label * 1
                    elif 'grocery_store' in audioname:
                        asc_label = asc_label * 2
                    elif 'metro_station' in audioname:
                        asc_label = asc_label * 3
                    elif 'residential_area' in audioname:
                        asc_label = asc_label * 4

                    batch_output = apply_model(model, feat.T, asc_label, 200, device)
                else:
                    batch_output = apply_model(model, feat.T, None, 200, device)
                # output for each file
                framewise_output = batch_output.squeeze().detach().cpu().numpy()
                eval_meta(result_folder, audioname+'.wav', framewise_output)

                # Append to evaluate the whole test fold at once
                if nbatch == 0:
                    fold_target = hard_label
                    fold_output = framewise_output
                
                    
                else:
                    fold_target = np.append(fold_target, hard_label, axis=0)
                    fold_output = np.append(fold_output, framewise_output, axis=0)
                    
                if nbatch_ == 0:
                    fold_single_target = hard_label
                    fold_single_output = framewise_output
                else:
                    fold_single_target = np.append(fold_single_target, hard_label, axis=0)
                    fold_single_output = np.append(fold_single_output, framewise_output, axis=0)
                
                
                nbatch += 1
                nbatch_ += 1

        ## evaluate single fold    
        reference_ = process_event(config.labels_hard, fold_single_target.T, 
                                config.posterior_thresh,
                                config.hop_len / config.audio_sr)

        results_ = process_event(config.labels_hard, fold_single_output.T, 
                                config.posterior_thresh,
                                config.hop_len / config.audio_sr)
        
        segment_based_metrics_fold.evaluate(
            reference_event_list=reference_,
            estimated_event_list=results_
        )
        
        single_segment_based_metrics_ER = segment_based_metrics_fold.overall_error_rate()
        single_segment_based_metrics_f1 = segment_based_metrics_fold.overall_f_measure()
        f1_single_1sec_list = single_segment_based_metrics_f1['f_measure']
        er_single_1sec_list = single_segment_based_metrics_ER['error_rate']
        print(f'\nMicro segment based metrics - ER: {round(er_single_1sec_list,3)} F1: {round(f1_single_1sec_list*100,2)} ')
        class_wise_metrics_ = segment_based_metrics_fold.results_class_wise_metrics()
        macroFs_ = []
        for c_ in class_wise_metrics_:
            macroFs_.append(class_wise_metrics_[c_]["f_measure"]["f_measure"])
        print(f'\nMacro segment based F1: {round((sum(np.nan_to_num(macroFs_))/11)*100,2)} ')
        segment_based_metrics_fold.reset()
        
        
        del model,fold_single_target,fold_single_output
        
    reference = process_event(config.labels_hard, fold_target.T, 
                            config.posterior_thresh,
                            config.hop_len / config.audio_sr)

    results = process_event(config.labels_hard, fold_output.T, 
                            config.posterior_thresh,
                            config.hop_len / config.audio_sr)

    segment_based_metrics_test.evaluate(
        reference_event_list=reference,
        estimated_event_list=results
    )

    overall_segment_based_metrics_ER = segment_based_metrics_test.overall_error_rate()
    overall_segment_based_metrics_f1 = segment_based_metrics_test.overall_f_measure()
    f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
    er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']
    print('************** overrall **************')
    print(f'\nMicro segment based metrics - ER: {round(er_overall_1sec_list,3)} F1: {round(f1_overall_1sec_list*100,2)} ')
    class_wise_metrics = segment_based_metrics_test.results_class_wise_metrics()
    macroFs = []
    for c in class_wise_metrics:
        macroFs.append(class_wise_metrics[c]["f_measure"]["f_measure"])
    print(f'\nMacro segment based F1: {round((sum(np.nan_to_num(macroFs))/11)*100,2)} ')
    segment_based_metrics_test.reset()

    # ========== all folds ================= #
    if 1:
        print('\n')
        path_groundtruth = config.path_groundtruth
        # Calculate PSDS SED metrics
        get_PSDS(path_groundtruth, result_folder)


