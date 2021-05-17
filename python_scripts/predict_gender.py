'''
Author/year: Rajat Hebbar, 2019

Predict frame-level gender of an audio segment (agnostic of speech) using
vggish-embeddings extracted for segments of length 0.96s
     
Input
    1) expt_dir   : Directory in which to write all output files
    2) model_file : pre-trained Keras (TF backend) model to predict gender 
    3) overlap    : fraction overlap of segments during inference of gender ID labels

Output
    1) timestamps-file : '.ts' ext file for each input file consisting of SAD segments 
                         labelled with gender ID in the following format:
                        <start-time-in-sec>\t<end_time-in-sec>\t<M/F>
    2) posterior-file  : '.post' ext file for each input file consisting of gender ID 
                         posteriors (probability of female speaker)
                        from the model at frame level i.e., 0=M, 1=F at 100 fps

Usage
   python predict_gender.py [expt_dir] [model_file] [overlap]

Example
   python predict_gender.py gender_out_dir models/gender.h5 0.5

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
import warnings
warnings.filterwarnings("ignore")
import argparse

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

def resegment(segments):
    data = {'male': [x for x in segments if x[-1] == 'M'], \
           'female': [x for x in segments if x[-1] == 'F']}
    
    seg_data = []
    for gender in ['male', 'female']:
        idx=0
        while idx < len(data[gender]):
            start = data[gender][idx][0]
            end = data[gender][idx][1]
            if idx != len(data[gender])-1:
                while data[gender][idx+1][0] == data[gender][idx][1]:
                    idx += 1
                    end = data[gender][idx][1]
                    if idx == len(data[gender])-1: break
            seg_data.append([start, end, gender])        
            idx += 1 
    
    return sorted(seg_data)

def main():
    parser = argparse.ArgumentParser(description='Predict frame-level gender of an audio segment')
    parser.add_argument('-o', '--overlap', type=float, metavar='overlap', help='fraction overlap of segments during inference of gender ID labels')
    parser.add_argument('expt_dir', type=str, help='Directory in which to write all output files')
    parser.add_argument('model_file', type=str, help='pre-trained Keras (TF backend) model to predict gender')
    parser.add_argument('nj', type=int, help='number of parallel threads')
    args = parser.parse_args()
    
    config=tf.ConfigProto(intra_op_parallelism_threads=args.nj, inter_op_parallelism_threads=args.nj)
    gender_seg_len = 0.96   # Feature segment length for gender ID
    effective_seg_len = (1-args.overlap)*gender_seg_len
    gender = {'0':'M','1':'F'}
    
    sad_ts_dir = os.path.join(args.expt_dir, 'SAD/timestamps/')
    write_post = os.path.join(args.expt_dir, 'GENDER/posteriors/')
    write_ts   = os.path.join(args.expt_dir, 'GENDER/timestamps/')
    feats_path = os.path.join(args.expt_dir, 'features/vggish/')
    
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
        model = load_model(args.model_file)
        
        while 1:
            try:
                [movie, feats] = sess.run([tf_key, tf_fts])
                pred_out = model.predict(feats)
                total_len = feats.shape[0] * effective_seg_len + args.overlap * gender_seg_len
                pred_frame_level = np.zeros(int(100*total_len))
                
                ## Assign labels to overlapped segments at frame level (100fps)
                for seg_num in range(feats.shape[0]):
                    seg_start = int(seg_num * effective_seg_len * 100)
                    seg_end = int((seg_num+1) * effective_seg_len * 100)
                    pred_frame_level[seg_start:seg_end] = pred_out[seg_num][-1]
                
                pred_frame_level[seg_end:] = pred_out[-1][-1]
                    
                
                with open(os.path.join(write_post, movie + '.post'),'w') as post_fp:
                    for label in pred_frame_level:
                        post_fp.write('{0:0.2f}\n'.format(label))
                
                sad_data = [x.rstrip().split() for x in open(os.path.join(sad_ts_dir, movie + '_subsegments.ts'), 'r').readlines()]
                sad_times = [[float(x[-2]), float(x[-1])] for x in sad_data]
                gender_labels = np.round(pred_frame_level)
                
                fts = open(os.path.join(write_ts, movie + '.ts'),'w')

                seg_data = []
                for seg in sad_times:
                    start = int(seg[0]*100)
                    end = int(seg[1]*100)
                    if start>=end or end > len(gender_labels):
                        continue
                    gender_label_seg = int(np.median(gender_labels[start:end]))
                    seg_data.append([seg[0], seg[1], gender[str(gender_label_seg)]])

                resegmented_ts_data = resegment(seg_data)
                with open(os.path.join(write_ts, movie + '.ts'), 'w') as ts_fp:
                    ts_fp.write('\n'.join([' '.join([str(x) for x in seg]) for seg in resegmented_ts_data]))
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    main()
