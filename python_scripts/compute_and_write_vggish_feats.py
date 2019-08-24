###
###
###     Modified version of AudioSet feature extraction 
###     script. Extract 128-dimensional embeddings of 
###     non-overlapping 0.96s audio segments. 
###     
###     Arguments:
###         expt_dir    -   Project directory where main script is executed.
###         wav_file    -   path to single audio file.
###         write_dir   -   Directory in which output files are stored.
###

""" A simple demonstration of running VGGish in inference mode.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

"""

from __future__ import print_function
import os, sys, numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from scipy.io import wavfile

config=tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)

proj_dir, wav_file, write_dir = sys.argv[1:-1]
overlap = float(sys.argv[-1])

AS_dir = os.path.join(proj_dir,'python_scripts/audioset_scripts/')
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
  vggish_params.EXAMPLE_HOP_SECONDS = (1-overlap)*vggish_params.EXAMPLE_WINDOW_SECONDS

  # If needed, prepare a record writer_dict to store the postprocessed embeddings.

  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    movie_id = wav_file[wav_file.rfind('/')+1:wav_file.rfind('.')]

    examples_batch = vggish_input.wavfile_to_examples(wav_file)
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
    writer = tf.python_io.TFRecordWriter(os.path.join(write_dir, movie_id + '.tfrecord'))
    writer.write(seq_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
  if not os.path.exists(write_dir):
    os.makedirs(write_dir)
  tf.app.run()
