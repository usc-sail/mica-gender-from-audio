# mica-gender-from-audio
Generate gender and SAD timestamps of audio based on neural network models trained in Keras. 
Input must be a text file containing full paths to either mp4/mkv media files or .wav audio 
files, and optionally the path to the directory where all the output will be 
stored (default=$proj_dir/expt).  
Outputs will be a text file for each movie/audio file, each line of which will contain the 
start and end times for the speech segment followed by the gender (male/female).
Frame level posteriors are also saved in the output directory.

## Usage: 
    bash generate_gender_timestamps.sh [-h] [-c config_file] movie_paths.txt (out_dir)  
    e.g.: bash generate_gender_timestamps.sh -c ./config.sh demo.txt DEMO  
    where:   
    -h              : Show help 
    -c              : Configuration file
    movie_paths.txt : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line    
    out_dir         : Directory in which to store all output files (default: "$PWD/gender_out_dir")  

## Example config file:
    nj=4                # Number of files to process simultaneously 
    feats_flag="y"      # "y"/"n" flag to keep kaldi-feature files after inference
    wavs_flag="n"       # "y"/"n" flag to keep .wav audio files after inference
    sad_overlap=0       # % overlap in SAD-segments (range: 0-1, 0 for no overlap) (single segment is 0.64s)
    gender_overlap=0    # % overlap in GENDER-segments (single segment is 0.96s) 
    uniform_seg_len=2.0   # Segment length for uniform speaker-segmentation 

##  Dependencies :
    kaldi                    :   ensure that all kaldi binaries are added to system path. If not,
                                     either add them to system path, or modify kaldi_root in 1st line of
                                     'path.sh' to reflect kaldi installation directory.
    keras, tensorflow        :   required to load data, model and make VAD predictions.
    Other python libraries required include numpy, scipy, resampy.


This tool can be used for noise-robust gender identification from audio. Two parallel systems are implemented for this purpose: 
1. Speech Activity Detection (SAD), and  
2. Gender Identification (GID) of speech segments.  

Both of the DNN-based systems make predictions at segment-level as opposed to traditional frame-level analysis. Segment duration for the SAD system is 0.64s (design choice) and for the GID system is 0.96s (pre-trained [*VGGish*](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) embeddings). For more details about the architecture and training procedures, please refer to the ICASSP '19 [paper](https://ieeexplore.ieee.org/document/8682532) (SAD), and INTERSPEECH '18 [paper](https://ieeexplore.ieee.org/abstract/document/8682532) (GID).

