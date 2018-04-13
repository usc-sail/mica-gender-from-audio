import numpy as np
import requests
import os
import sys
from scipy import signal as sig
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#from train_gender import FullyConnected
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2


def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)


def generate_single_example(example):
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
    vad_ts_dir = expt_dir + '/VAD/timestamps/'
    write_post = expt_dir + '/GENDER/posteriors/'
    write_ts = expt_dir + '/GENDER/timestamps/'
    feats_path = sys.argv[2] + '/'

    if not os.path.exists(write_post):
        os.makedirs(write_post)
    if not os.path.exists(write_ts):
        os.makedirs(write_ts)
    pwd = os.getcwd()
    if not os.path.isfile(pwd + '/python_scripts/audioset_scripts/vggish_model.ckpt')
        download_file_from_google_drive('1c-wi6F_Fv0Z0TmJBpSrlTT0iCDmKF_NJ', pwd + '/python_scripts/audioset_scripts/vggish_model.ckpt')
    tfr_file = os.listdir(feats_path)
    tfr_paths = [feats_path + x for x in tfr_file]
    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer(tfr_paths, num_epochs=1, shuffle=False)
    with tf.Session(config=config) as sess:
        K.set_session(sess)
        _, ser_example = reader.read(file_queue)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        model = load_model(sys.argv[3])

        used = []

        while 1:
            try:
                context, sequence = generate_single_example(ser_example)
                [movie, feats] = sess.run([context, sequence])
                pred = model.predict(feats)
                fpost = open(write_post + movie + '.post','w')
                for label in pred:
                    fpost.write('{0:0.2f}\n'.format(label[1]))
                fpost.close()
                
                fts = open(write_ts + movie + '.ts','w')
                vad_data = [x.rstrip().split() for x in open(vad_ts_dir + movie + '.ts', 'r').readlines()]
                vad_times = [[float(x[0]), float(x[1])] for x in vad_data]
                gender_labels = sig.medfilt(np.round([x[1] for x in pred]), 3)
                gender_labels = np.round([np.repeat(x,96) for x in gender_labels]).flatten()
                for seg in vad_times:
                    start = int(seg[0]/0.01)
                    end = int(seg[1]/0.01)
                    if start>=end:
                        continue
                    gender_seg = gender_labels[start:end]
                    diff = np.diff(gender_seg)
                    seg_start = [(ind+1) for ind in xrange(len(diff)) if diff[ind]==1]
                    seg_end = [ind for ind in xrange(len(diff)) if diff[ind]==-1]
                    seg_times = list(zip(seg_start, seg_end))
                    if len(seg_times)==0:
                        gender_seg = int(np.median(gender_seg))
                        gender = {'0':'M','1':'F'}
                        fts.write('{}\t{}\t{}\n'.format(seg[0],seg[1],gender[str(gender_seg)]))
                    else:
                        if seg_times[0][0]>0:
                            end_gen = seg[0] + seg_times[0][0]*0.01
                            fts.write('{}\t{}\t{}\n'.format(seg[0], end_gen, 'M'))
                        for idx,gen_seg in enumerate(seg_times):
                            start_gen = seg[0] + gen_seg[0]*0.01
                            end_gen = seg[0] + gen_seg[1]*0.01
                            fts.write('{}\t{}\t{}\n'.format(start_gen,end_gen,'F'))
                            start_gen = end_gen + 0.01
                            if idx==len(seg_times)-1:
                                end_gen = seg[1]
                            else:
                                end_gen = seg[0] + seg_times[idx+1][0]*0.01 - 0.01
                            fts.write('{}\t{}\t{}\n'.format(start_gen, end_gen, 'M'))
                fts.close()
            except:
                coord.request_stop()
                coord.join(threads)
                sess.close()
                break
if __name__ == '__main__':
    main()
