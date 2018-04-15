# mica-gender-from-audio
Generate gender and VAD timestamps of audio based on neural network models trained on keras. 
Input must be a text file containing full paths to either mp4/mkv files or wav 
files, and optionally the path to the directory where all the output will be 
stored (default=$proj_dir/expt).  
Outputs will be a text file for each movie/audio file, each line of which will contain the 
start and end times for the speech segment followed by the gender (M/F).
Frame level posteriors are also saved in the output directory.

## Usage: 
    generate_gender_timestamps.sh [-h] [-w y/n] [-f y/n] [-j num_jobs] movie_paths.txt (out_dir)  
    e.g.: generate_gender_timestamps.sh -w y -f y -j 8 demo.txt DEMO  
    where:   
    -h              :  Show help   
    -w              :  Store wav files after processing (default: n)  
    -f              :  Store feature files after processing (default: n)   
    -j              :  Number of parallel jobs to run (default: 16)   
    movie_paths.txt : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line    
    out_dir         : Directory in which to store all output files (default: "$PWD/gender_out_dir")  


##  Packages/libraries required :
    kaldi                    :   ensure that all kaldi binaries are added to system path. If not,
                                     either add them to system path, or modify kaldi_root in 1st line of
                                     'path.sh' to reflect kaldi installation directory.
    keras, tensorflow-gpu    :   required to load model and make VAD predictions.
    re, gzip, struct         :   used by kaldi_io, to read feature files.
    Other python libraries required include numpy, scipy, resampy.


This tool can be used for noise-robust gender identification from audio. Two parallel systems are implemented for this purpose: 
1. Voice Activity Detection (VAD), and  
2. Gender classification of speech segments.  

VAD is implemented using a recurrent neural-network (BLSTM) trained on movie audio, while gender classification is implemented using a deep neural network architecture trained on a subset of AudioSet dataset. The input to the VAD system are 23 dimensional  log-Mel filterbank features corresponding to a 10ms frame (with 15 frames context), and the input to the gender system is a pre-trained embedding (link to VGGish) corresponding to a 960ms segment. The gender segments, in the form of timestamps, are obtained via masking the gender predictions by the VAD labels obtained.
