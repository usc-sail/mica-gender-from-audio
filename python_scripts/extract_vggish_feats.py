from __future__ import print_function

'''
Modified version of AudioSet feature extraction script. 
Extract 128-dimensional embeddings of non-overlapping 0.96s audio segments. 
     
Input
    1) proj_dir    : project directory where main script is executed.
    2) wav_file    : path to single audio file.
    3) write_dir   : directory in which output files are stored.
    4) overlap     : fraction overlap of segments during inference of gender ID labels

Output 
    1) {file_id}.tfrecord   : VGGish features of 128-dimension 

Usage
    python extract_vggish_feats.py [proj_dir] [wav_file] [write_dir] [overlap]

Example
    python extract_vggish_feats.py . gender_out_dir/wavs/temp.wav gender_out_dir/features/vggish 0.5

'''

""" A simple demonstration of running VGGish in inference mode.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

"""

import os, sys, numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from scipy.io import wavfile
import argparse


parser = argparse.ArgumentParser(description='Extract 128D VGGish features')
parser.add_argument('-o','--overlap', type=float, metavar='overlap', help='fraction overlap of segments during inference of gender ID labels')
parser.add_argument('proj_dir', type=str, help='project directory where main script is executed')
parser.add_argument('wav_file', type=str, help='path to single audio file')
parser.add_argument('write_dir', type=str, help=' directory in which output files are stored')
parser.add_argument('nj', type=int, help='Number of parallel threads')
args = parser.parse_args()

config=tf.ConfigProto(intra_op_parallelism_threads=args.nj, inter_op_parallelism_threads=args.nj)
AS_dir = os.path.join(args.proj_dir,'python_scripts/audioset_scripts/')
pca_params = os.path.join(AS_dir,'vggish_pca_params.npz')
checkpoint= os.path.join(AS_dir,'vggish_model.ckpt')
sys.path.insert(0,AS_dir)

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim


def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(pca_params)
  vggish_params.EXAMPLE_HOP_SECONDS = (1-args.overlap)*vggish_params.EXAMPLE_WINDOW_SECONDS

  # If needed, prepare a record writer_dict to store the postprocessed embeddings.

  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    movie_files = [x.rstrip() for x in open(args.wav_file, 'r').readlines()] 

    for movie_file in movie_files:
        movie_id = movie_file[movie_file.rfind('/')+1:movie_file.rfind('.')]
        if type(movie_id) == str:
            movie_id = movie_id.encode()
        write_file = os.path.join(args.write_dir, movie_id + '.tfrecord') 
        if os.path.exists(write_file):
            continue

        examples_batch = vggish_input.wavfile_to_examples(movie_file)
        num_splits = min(int(examples_batch.shape[0]/10), 100)
        num_splits = max(1, num_splits)
        examples_batch = np.array_split(examples_batch, num_splits)

        embedding_batch = []
        for i in range(num_splits):
            [batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch[i]})
            embedding_batch.extend(batch)

        postprocessed_batch = pproc.postprocess(np.array(embedding_batch))

        # Write the postprocessed embeddings as a SequenceExample, in a similar
        # format as the features released in AudioSet. Each row of the batch of
        # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # the rows are written as a sequence of bytes-valued features, where each
        # feature value contains the 128 bytes of the whitened quantized embedding.
        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'movie_id':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[movie_id]))
                        }
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                        tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[embedding.tobytes()]))
                                for embedding in postprocessed_batch
                            ]
                        )
                }
            )
        )
        writer = tf.python_io.TFRecordWriter(write_file)
        writer.write(seq_example.SerializeToString())
        writer.close()

if __name__ == '__main__':
  if not os.path.exists(args.write_dir):
    os.makedirs(args.write_dir)
  tf.app.run()
