###
###     Python Script to generate VAD labels given spliced-
###     log-Mel features as input
###
###     INPUTS:
###     write_dir    -  Directory in which to write all output files
###     scp_file     -  Kaldi feature file in .scp format
###     model_file   -  VAD model file trained on keras
###

###
###     OUTPUTS:
###     Frame-level posteriors from the model predictions
###     are thresholded at 0.5 and median-filtered with window length
###     of 550ms.
###    
###     write_post   -  Raw posteriors representing confidence in VAD prediction
###     write_ts     -  VAD segments detected written as start and end 
###                     end times.
###
###

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


K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
write_dir, scp_file, model_file = sys.argv[1:]
frame_len = 0.01
vad_wav_dir = os.path.join(write_dir, 'VAD/wavs') 
write_post = os.path.join(write_dir, 'VAD/posteriors/')
write_ts   = os.path.join(write_dir, 'VAD/timestamps/')

model = load_model(model_file)
gen = rms(scp_file)

predictions = []
# Generate VAD posteriors using pre-trained VAD model
for key, mat in gen:
    mat = mat.reshape(mat.shape[0],16,23)
    pred = model.predict(mat, batch_size=50, verbose=0)
    pred = [x[1] for x in pred]
    predictions.extend(pred)
movie = key.split('_seg')[0]

# Post-processing of posteriors
labels = np.round(predictions)
labels_med_filt = sig.medfilt(labels, 55)
seg_times =frame2seg(labels_med_filt)

# Write start and end VAD timestamps 
fw = open(os.path.join(write_ts, movie + '_wo_ss.ts'),'w')
if not os.path.exists(os.path.join(vad_wav_dir,movie)):
    os.makedirs(os.path.join(vad_wav_dir, movie))

for seg_id, segment in enumerate(seg_times):
    if segment[1]-segment[0] > 0.05:
        fw.write('{0:0.2f}\t{1:0.2f}\n'.format(segment[0], segment[1]))
        ## 16kHz audio segments required to perform speaker homogenous segmentation
        cmd = 'sox -V1 {0}.wav -r 16k {1}/{2}_vad-{3:04}.wav trim {4} ={5}'.format(os.path.join(write_dir,'wavs',movie), os.path.join(vad_wav_dir, movie), movie, seg_id + 1, segment[0], segment[1])
        os.system(cmd)
fw.close()

# Write frame-level posterior probabilities
fpost = open(os.path.join(write_post, movie + '.post'),'w')
for frame in predictions:
    fpost.write('{0:0.2f}\n'.format(frame))
fpost.close()

fw.close()
fpost.close()
