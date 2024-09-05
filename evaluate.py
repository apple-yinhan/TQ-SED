import os
import numpy as np
import config
from dcase_util.containers import metadata
import sed_scores_eval
from sed_scores_eval import segment_based


def find_contiguous_regions(activity_array):
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]
    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, len(activity_array)]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))




def eval_meta(output_folder, audio_name, framewise_output):

    hop_length_seconds = config.hop_len / config.audio_sr
    timestamps = np.arange(0, framewise_output.shape[0] + 1) * hop_length_seconds

    sed_scores_eval.io.write_sed_scores(
        framewise_output, os.path.join(output_folder, audio_name +'.tsv'),
        timestamps=timestamps, event_classes=config.labels_hard
    )




def process_event(class_labels, frame_probabilities, threshold, hop_length_seconds):
    results = []
    for event_id, event_label in enumerate(class_labels):
        # Make sure that the evaluated labels are the ones that correspond to the hard labels
        if event_label in config.labels_hard:
            # Binarization
            event_activity = frame_probabilities[event_id, :] > threshold

            # Convert active frames into segments and translate frame indices into time stamps
            event_segments = np.floor(find_contiguous_regions(event_activity) * hop_length_seconds)

            # Store events
            for event in event_segments:
                results.append(
                    metadata.MetaDataItem(
                        {
                            'event_onset': event[0],
                            'event_offset': event[1],
                            'event_label': event_label
                        }
                    )
                )
    
    results = metadata.MetaDataContainer(results)

    # Event list post-processing
    results = results.process_events(minimum_event_length=None, minimum_event_gap=0.1)#0.1
    results = results.process_events(minimum_event_length=0.1, minimum_event_gap=None)
    return results



# calculate sed eval metrics
def metric_perbatch(segment_based_metrics, framewise_output, target):

    results = process_event(config.labels_soft, framewise_output.T, config.posterior_thresh, config.hop_size / config.sample_rate) 
    reference = process_event(config.labels_soft, target.T, config.posterior_thresh, config.hop_size / config.sample_rate)
    
    segment_based_metrics.evaluate(
        reference_event_list=reference,
        estimated_event_list=results
    )


    return segment_based_metrics



def get_PSDS(path_groundtruth, path_scores):
    # Calculate PSDS

    f_best, p_best, r_best, thresholds_best, stats_best = segment_based.best_fscore(
        scores=path_scores,
        ground_truth=path_groundtruth,
        audio_durations= '/mnt/nfs2/hanyin/LASS4SED/baseline-codes/task4b_mycodes/metadata/development_metadata.csv', 
        segment_length=1.,
        min_precision=0.,
        min_recall=0.0,
        beta=1.,
        time_decimals=30,
        num_jobs=8,
    )
    print(thresholds_best)
    print('-----------------------\n')
    for cls in f_best:
        print(cls)
        print(' ', 'f:', f_best[cls])
        print(' ', 'p:', p_best[cls])
        print(' ', 'r:', r_best[cls])
    print('*****************************\n')