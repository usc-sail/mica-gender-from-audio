'''
Author/year: Rajat Hebbar, 2019

Generate VAD labels using log-Mel features extracted for 0.64s segments as input. 

Input
    1) expt_dir     :  Directory in which to write all output files
    2) feats_scp    :  Kaldi feature file in .scp format
    3) model_file   :  VAD model file trained on keras
    4) overlap      :  fraction overlap of segments during VAD post-processing

Output
    1) timestamps-file   :  '.ts' ext file for each input file consisting of
                            SAD segments in the following format:
                            <seg-id> <file-id> <seg-start-time> <seg-end-time>
    2) posteriors-file   :  '.post' ext file for each input file consisting of
                            SAD posterior probability (of speech) at 100fps

Usage
    python generate_vad_labels.py [expt_dir] [feats_scp] [model_file] [overlap]

Example
    python generate_vad_labels.py gender_out_dir models/vad.h5 0.5

'''

from __future__ import division
import os, sys, numpy as np
np.warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.models import load_model
from scipy import signal as sig
from kaldi_io import read_mat_scp as rms
from keras import backend as K
import warnings
#warnings.filterwarnings("ignore")


##  Convert frame-level posteriors into 
##  continuous segments of regions where post=pos_label
def frame2seg(frames, frame_time_sec=0.01, pos_label=1):
    pos_idxs = np.where(frames==pos_label)[0]
    pos_regions = np.split(pos_idxs, np.where(np.diff(pos_idxs)!=1)[0]+1)
    if len(pos_idxs) == 0 or len(pos_regions) == 0:
        return []
    segments = np.array([[x[0], x[-1]+1] for x in pos_regions])*frame_time_sec
    return segments

## Mean and variance normalization for features
def normalize(data):
    return np.divide(np.subtract(data, np.mean(data)), np.std(data))


## Uniform segmentation of VAD segments
def perform_uniform_segmentation(movie, segments, write_dir, max_segment_duration=2, min_remaining_duration=0.64):
    write_file = os.path.join(write_dir, movie + '_subsegments.ts')
    subseg_fp = open(write_file, 'w')
    dur_threshold = max_segment_duration + min_remaining_duration

    for seg_num, seg in enumerate(segments):
        utt_id = '{movie}_vad-{seg_num:04d}'.format(movie=movie, seg_num=seg_num)
        start_time = seg[0]
        end_time = seg[1]

        dur = end_time - start_time

        start = start_time
        while (dur > dur_threshold):
            end = start + max_segment_duration
            start_relative = start - start_time
            end_relative = end - start_time
            new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                utt_id=utt_id, s=int(100 * start_relative),
                e=int(100 * end_relative))
            subseg_fp.write("{new_utt} {utt_id} {s:.3f} {e:.3f}\n".format(
                new_utt=new_utt, utt_id=utt_id, s=start,
                e=start + max_segment_duration))
            start += max_segment_duration 
            dur -= max_segment_duration

        end = end_time
        new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
            utt_id=utt_id, s=int(round(100 * (start - start_time))),
            e=int(round(100 * (end - start_time))))
        subseg_fp.write("{new_utt} {utt_id} {s:.3f} {e:.3f}\n".format(
            new_utt=new_utt, utt_id=utt_id, s=start,
            e=end))
    subseg_fp.close()        


def main():
    expt_dir, feats_scp, model_file = sys.argv[1:-2]
    overlap = float(sys.argv[-2])
    uniform_seg_len = float(sys.argv[-1])
    assert overlap >=0 and overlap <1, "Invalid choice of overlap, should be 0 <= overlap < 1"
    num_frames, num_freq_bins = (64, 64)
    fps = 100
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    shift = num_frames - int(overlap*num_frames)
    vad_wav_dir = os.path.join(expt_dir, 'VAD/wavs') 
    write_post = os.path.join(expt_dir, 'VAD/posteriors/')
    write_ts   = os.path.join(expt_dir, 'VAD/timestamps/')

    model = load_model(model_file)
    gen = rms(feats_scp)

    # Generate VAD posteriors using pre-trained VAD model
    for movie, fts in gen:
        num_seg = int((len(fts) - num_frames) // shift)
        pred = [ [] for _ in range(fts.shape[0]) ]
       #num_seg = int(len(fts)//64)
        for i in range(num_seg):
            feats_seg = normalize(fts[i*shift : i*shift + num_frames])
            p = model.predict(feats_seg.reshape((1, num_frames, num_freq_bins, 1)), verbose=0)
            for j in range(i*shift, i*shift + num_frames):
                pred[j].extend([p[0][1]])
        predictions = np.array([np.median(pred[i]) if pred[i]!=[] else 0 for i in range(fts.shape[0])])

        # Post-processing of posteriors
        labels = np.round(predictions)
        seg_times = frame2seg(np.round(labels))
        perform_uniform_segmentation(movie, seg_times, write_ts, max_segment_duration=uniform_seg_len, min_remaining_duration=0.64)

        # Write start and end VAD timestamps 
        with open(os.path.join(write_ts, movie + '.ts'),'w') as ts_fp:
            for seg_num, segment in enumerate(seg_times):
                ts_fp.write('{0}_vad-{1:04} {0} {2:0.2f} {3:0.2f}\n'.format(movie, seg_num, segment[0], segment[1]))

        # Write frame-level posterior probabilities
        with open(os.path.join(write_post, movie + '.post'),'w') as post_fp:
            for frame in predictions:
                post_fp.write('{0:0.2f}\n'.format(frame))

if __name__ == '__main__':
    main()

