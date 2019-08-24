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


##
##  Convert frame-level posteriors into 
##  continuous segments of regions where post=pos_label
##
def frame2seg(frames, frame_time_sec=0.01, pos_label=1):
    pos_idxs = np.where(frames==pos_label)[0]
    pos_regions = np.split(pos_idxs, np.where(np.diff(pos_idxs)!=1)[0]+1)
    if len(pos_idxs) == 0 or len(pos_regions) == 0:
        return []
    segments = np.array([[x[0], x[-1]+1] for x in pos_regions])*frame_time_sec
    return segments

def normalize(data):
    return np.divide(np.subtract(data, np.mean(data)), np.std(data))

def main():
    expt_dir, feats_scp, model_file = sys.argv[1:-1]
    overlap = float(sys.argv[-1])
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
        
        # Write start and end VAD timestamps 
        fw = open(os.path.join(write_ts, movie + '_wo_ss.ts'),'w')
        if not os.path.exists(os.path.join(vad_wav_dir,movie)):
            os.makedirs(os.path.join(vad_wav_dir, movie))

        seg_ct = 1
        for segment in seg_times:
        # Threshold to 50ms minimum segment duration
            if segment[1]-segment[0] > 0.05:
                fw.write('{0}_vad-{1:04} {0} {2:0.2f} {3:0.2f}\n'.format(movie, seg_ct, segment[0], segment[1]))
                seg_ct += 1
        fw.close()

        # Write frame-level posterior probabilities
        fpost = open(os.path.join(write_post, movie + '.post'),'w')
        for frame in predictions:
            fpost.write('{0:0.2f}\n'.format(frame))
        fpost.close()

        fw.close()

if __name__ == '__main__':
    main()

