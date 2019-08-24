'''
Author/year: Rajat Hebbar, 2018

For model evaluations and baseline, see: 

Predict frame-level gender of an audio segment (agnostic of speech) using
vggish-embeddings extracted for segments of length 0.96s
     
Input:
    1) expt_dir   : parent directory which contains 'features' and 'VAD' directories
    2) model_file : pre-trained Keras (TF backend) model to predict gender for a
                    segment
    3) overlap    : fraction overlap of features during inference of gender ID labels

Output:
    1) timestamps-file : writes out a .ts file for each wav file with the following format
                        <start-time-in-sec>\t<end_time-in-sec>\t<M/F>
    2) posterior-file  : predicted gender posterior (probability of female speaker)
                        from the model at frame level i.e., 0=M, 1=F

Usage:
    

Example

'''
import numpy as np
np.warnings.filterwarnings('ignore')
import os, sys, glob 
from scipy import signal as sig
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
config=tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
import warnings
warnings.filterwarnings("ignore")
#from train_gender import FullyConnected
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2

def feature_parser(example):
    context_features = {'movie_id': tf.FixedLenFeature([], tf.string)}
    sequence_features = {'audio_embedding': tf.FixedLenSequenceFeature([], tf.string)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(example, 
        context_features = context_features, sequence_features = sequence_features)

    normalized_feature = tf.divide(
                tf.decode_raw(sequence_parsed['audio_embedding'], tf.uint8),
                tf.constant(255, tf.uint8))
    shaped_feature = tf.reshape(tf.cast(normalized_feature, tf.float32),
                                    [-1, 128])
    
    return context_parsed['movie_id'], shaped_feature


def main():
    expt_dir = sys.argv[1]
    overlap = float(sys.argv[-1])
    fps = 100       # Used to project overlapped segment-level decisions to frame-level 
    gender_seg_len = 0.96   # Feature segment length for gender ID
    effective_seg_len = (1-overlap)*gender_seg_len
    gender = {'0':'M','1':'F'}
    
    vad_ts_dir = os.path.join(expt_dir, 'VAD/timestamps/')
    write_post = os.path.join(expt_dir, 'GENDER/posteriors/')
    write_ts   = os.path.join(expt_dir, 'GENDER/timestamps/')
    feats_path = os.path.join(expt_dir, 'features/vggish/')
    
    if not os.path.exists(write_post):  os.makedirs(write_post)
    if not os.path.exists(write_ts): os.makedirs(write_ts)

    with tf.Session(config=config) as sess:
        # Prepare TF Dataset for inference
        tfr_files = glob.glob(os.path.join(feats_path, '*.tfrecord'))
        dataset = tf.data.TFRecordDataset(tfr_files)
        dataset = dataset.map(feature_parser)
        dataset_itr = dataset.make_one_shot_iterator()
        tf_key, tf_fts = dataset_itr.get_next()

        K.tensorflow_backend.set_session(sess)
        model = load_model(sys.argv[2])
        
        while 1:
            try:
                [movie, feats] = sess.run([tf_key, tf_fts])
                pred_out = model.predict(feats)
                total_len = feats.shape[0] * effective_seg_len + overlap * gender_seg_len
                pred_frame_level = np.zeros(int(100*total_len))
                
                ## Assign labels to overlapped segments at frame level (100fps)
                for seg_num in range(feats.shape[0]):
                    seg_start = int(seg_num * effective_seg_len * 100)
                    seg_end = int((seg_num+1) * effective_seg_len * 100)
                    pred_frame_level[seg_start:seg_end] = pred_out[seg_num][-1]
                
                pred_frame_level[seg_end:] = pred_out[-1][-1]
                    
                
                fpost = open(os.path.join(write_post, movie + '.post'),'w')
                for label in pred_frame_level:
                    fpost.write('{0:0.2f}\n'.format(label))
                fpost.close()
                
                vad_data = [x.rstrip().split() for x in open(os.path.join(vad_ts_dir, movie + '.ts'), 'r').readlines()]
                vad_times = [[float(x[0]), float(x[1])] for x in vad_data]
                gender_labels = np.round(pred_frame_level)
                
                fts = open(os.path.join(write_ts, movie + '.ts'),'w')

                for seg in vad_times:
                    start = int(seg[0]*100)
                    end = int(seg[1]*100)
                    if start>=end or end > len(gender_labels):
                        continue
                    gender_seg = int(np.median(gender_labels[start:end]))
                    fts.write('{}\t{}\t{}\n'.format(seg[0], seg[1], gender[str(gender_seg)]))
                fts.close()
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    main()
