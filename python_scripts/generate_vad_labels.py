###
###     Python Script to generate VAD labels given spliced-
###     log-Mel features as input
###
###     INPUTS:
###     frame_len    -  Length of single feature-frame in seconds
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

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
write_dir, scp_file, model_file = sys.argv[1:]
frame_len = 0.01
write_post = os.path.join(write_dir, 'VAD/posteriors/')
write_ts   = os.path.join(write_dir, 'VAD/timestamps/')

model = load_model(model_file)
gen = rms(scp_file)

predictions = []
for key, mat in gen:
    mat = mat.reshape(mat.shape[0],16,23)
    pred = model.predict(mat, batch_size=50, verbose=0)
    pred = [x[1] for x in pred]
    predictions.extend(pred)
movie = key.split('_seg')[0]

labels = np.round(predictions)
labels_med_filt = sig.medfilt(labels, 55)
diff = np.diff(labels_med_filt)
seg_start = [(ind+1)*frame_len for ind in xrange(len(diff)) if diff[ind]==1]
seg_end = [ind*frame_len for ind in xrange(len(diff)) if diff[ind]==-1]
seg_times = list(zip(seg_start, seg_end))

fw = open(os.path.join(write_ts, movie + '.ts'),'w')
for segment in seg_times:
    fw.write('{0:0.2f}\t{1:0.2f}\n'.format(segment[0], segment[1]))
fw.close()

fpost = open(os.path.join(write_post, movie + '.post'),'w')
for frame in predictions:
    fpost.write('{0:0.2f}\n'.format(frame))
fpost.close()

fw.close()
fpost.close()
