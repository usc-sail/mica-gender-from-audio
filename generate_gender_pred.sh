#!/bin/bash
#. ./path.sh

##
##
##
##  Generate gender and VAD timestamps based on fbank-BLSTM model trained on keras. 
##  Input must be a text file containing full paths to either mp4/mkv files or wav 
##  files, and optionally the path to the directory where all the output will be 
##  stored (default=$proj_dir/expt)
##  Output will be a ctm file for each movie, each line of which will contain the 
##  start and end times for the speech segment, along with the gender.
##  Frame level posteriors are also saved
##  E.g., ./generate_gender_timestamps.sh movie_path_list.txt expt_dir
##
##  Variables to be aware of:
##      nj            : number of jobs to be run in parallel 
##      expt_dir      : directory where all the output-files will be created
##      
##  
##  Packages/libraries required :
##     kaldi          : ensure that all kaldi binaries are added to system path. If not,
##                      either add them to system path, or modify kaldi_root in 1st line of
##                      'path.sh' to reflect kaldi installation directory.
##     keras, tensorflow-gpu          :     required to load model and make VAD predictions.
##     re, gzip, struct               :     used by kaldi_io, to read feature files.
##     Other python libraries required include numpy, scipy, resampy.
##
##
##


usage="Script to identify gender from audio
Usage: $(basename "$0") [-h] [-w y/n] [-f y/n] [-j num_jobs] movie_paths.txt (out_dir)
e.g.: $(basename "$0") -w y -f y -nj 8 demo.txt DEMO

where:
-h                  : Show help 
-w                  : Store wav files after processing (default: n)
-f                  : Store feature files after processing (default: n)
-j                 : Number of parallel jobs to run (default: 16)
movie_paths.txt     : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line 
out_dir             : Directory in which to store all output files (default: "\$PWD"/gender_out_dir)
"

## Add kaldi binaries to path if path.sh file exists
if [ -f path.sh ]; then . ./path.sh; fi
## Default Options
feats_flag="n"
wavs_flag="n"
nj=16

## Input Options
while getopts ":hw:f:j:" option
do
case "${option}"
in
h) echo "$usage"
    exit;;
f) feats_flag="${OPTARG}";;
w) wavs_flag="${OPTARG}";;
j) nj=${OPTARG};;
\?) echo "Invalid option: -$OPTARG" >&2 
printf "See below for usage\n\n"
echo "$usage"
exit ;;
esac
done

## Input Arguments
movie_list=${@:$OPTIND:1}
exp_id=$(($OPTIND+1))
if [ $# -ge $exp_id ]; then
    expt_dir=${@:$exp_id:1}
else
    expt_dir=gender_out_dir
fi

#printf "nj=$nj\nexpt_dir=$expt_dir\nmovie_list=$movie_list\n"

proj_dir=$PWD
cmd="run.pl"
wav_dir=$expt_dir/wav_segments
full_wav_dir=$expt_dir/wavs
feats_dir=$expt_dir/features
scpfile=$feats_dir/wav.scp
lists_dir=$feats_dir/scp_lists
py_scripts_dir=$proj_dir/python_scripts
vad_model=$PWD/models/vad.h5
gender_model=$PWD/models/gender.h5
mkdir -p $wav_dir $full_wav_dir $feats_dir/log $lists_dir $expt_dir/VAD/timestamps $expt_dir/VAD/posteriors $expt_dir/GENDER/timestamps $expt_dir/GENDER/posteriors

### Create .wav files given movie_files
echo " >>>> CREATING WAV FILES <<<< "
bash_scripts/create_wav_files.sh $movie_list $expt_dir $nj
bash_scripts/create_scp.sh $wav_dir $scpfile

### Extract fbank-features
echo " >>>> EXTRACTING FEATURES FOR VAD <<<< "
bash_scripts/create_spliced_fbank_feats.sh $feats_dir $nj

## Generate VAD Labels
echo " >>>> GENERATING VAD LABELS <<<< "
movie_count=1
for movie_path in `cat $movie_list`
do
    movieName=`basename $movie_path | awk -F '.' '{print $1}'`
    cat $feats_dir/spliced_feats.scp | grep -- "${movieName}_seg" > $lists_dir/${movieName}_feats.scp
    python $py_scripts_dir/generate_vad_labels.py $expt_dir $lists_dir/${movieName}_feats.scp $vad_model &
    if [ $(($movie_count % $nj)) -eq 0 ];then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

### Create VGGish embeddings
echo " >>>> CREATING VGGISH EMBEDDINGS <<<< "
## Download vggish_model.ckpt file if not exists in python_scripts/audioset_scripts/ directory
python $py_scripts_dir/download_vggish_ckpt_file.py python_scripts/audioset_scripts/vggish_model.ckpt
#if ! [ -f $py_scripts_dir/audioset_scripts/vggish_model.ckpt ]; then
#    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1c-wi6F_Fv0Z0TmJBpSrlTT0iCDmKF_NJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c-wi6F_Fv0Z0TmJBpSrlTT0iCDmKF_NJ" -O $py_scripts_dir/audioset_scripts/vggish_model.ckpt && rm -rf /tmp/cookies.txt
#fi

ls $full_wav_dir/*.wav  > $expt_dir/wav.list
movie_count=1
for wav_file in `cat $expt_dir/wav.list`
do
    python $py_scripts_dir/compute_and_write_vggish_feats.py $proj_dir $wav_file $feats_dir/vggish &
    if [ $(($movie_count % $nj)) -eq 0 ]; then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

### Make gender predictions
echo " >>>> PREDICTING GENDER SEGMENTS <<<< "
python $py_scripts_dir/predict_gender_median.py $expt_dir $gender_model

## Delete feature files and/or wav files unless otherwise specified
if [[ "$feats_flag" == "n" ]]; then
    rm -r $feats_dir &
fi
if [[ "$wavs_flag" == "n" ]]; then
    rm -r $full_wav_dir $wav_dir &
fi
rm $expt_dir/wav.list 
wait
echo " >>>> GENDER SEGMENTS PER-MOVIE CAN BE FOUND IN $expt_dir/GENDER/timestamps <<<< "

