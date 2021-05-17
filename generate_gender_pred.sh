#!/bin/bash
#. ./path.sh

##
##
##
##  Generate gender and SAD timestamps based on models trained on keras. 
##  Input must be a text file containing full paths to either mp4/mkv files or wav 
##  files, and optionally the path to the directory where all the output will be 
##  stored (default=$proj_dir/gender_out_dir)
##  Output will be a timestamps file for each movie, each line of which will contain the 
##  start and end times for the speech segment, along with the gender.
##  Frame level posteriors are also saved if required
##  E.g., ./generate_gender_pred.sh movie_path_list.txt expt_dir
##
##  Variables to be aware of:
##      out_dir       : directory where all the output-files will be created
##  
##  Read the default config file (./config.sh) for more details about user-defined variables
##  for inference.
##
##  Packages/libraries required :
##     kaldi          : ensure that all kaldi binaries are added to system path. If not,
##                      either add them to system path, or modify kaldi_root in 1st line of
##                      'path.sh' to reflect kaldi installation directory.
##     keras, tensorflow              :     required to load model and make SAD predictions.
##     re, gzip, struct               :     used by kaldi_io, to read feature files.
##     Other python libraries required include numpy, scipy, resampy.
##
##
##

## Define usage of script
usage="Perform gender identification from audio
Usage: bash $(basename "$0") [-h] [-c config_file] movie_paths.txt (out_dir)
e.g.: bash $(basename "$0") -c config.sh demo.txt DEMO

where:
-h                  : Show help 
-c                  : Config file (default: ./config.sh)
movie_paths.txt     : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line 
out_dir             : Directory in which to store all output files (default: "\$PWD"/gender_out_dir)
"

## Kill all background processes if file exits due to error
trap "exit" INT TERM
trap "kill 0" EXIT
## Add kaldi binaries to path if path.sh file exists
if [ -f path.sh ]; then . ./path.sh; fi

## Default config file
config_file=./config.sh

## Input Options
if [ $# -eq 0 ];
then
    echo "$usage"
    exit
fi

while getopts ":hc:" option
do
    case "${option}"
    in
        h) echo "$usage"
        exit;;
        c) config_file=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG" >&2 
        printf "See below for usage\n\n"
        echo "$usage"
        exit ;;
    esac
done

## Import configurations file
if [ -f $config_file ]; then 
    . $config_file
    echo -e "\nUsing configuration :"
    paste $config_file | cut -f1 -d' '
else
    echo -e "\nNo config file found, pls check if default config.sh is present in your working directory"
fi

## Input Arguments
movie_list=${@:$OPTIND:1}
exp_id=$(($OPTIND+1))
if [ $# -ge $exp_id ]; then
    expt_dir=${@:$exp_id:1}
else
    expt_dir=gender_out_dir
fi

## Reduce nj if not enough files
num_movies=`cat $movie_list | wc -l`
if [ $num_movies -lt $nj ]; then
    nj=$num_movies
fi

proj_dir=$PWD
sad_model=$proj_dir/models/sad.h5
#sad_model=/proj/rajatheb/VAD/speech_enhancement/scripts/train_scripts/git/mica-speech-activity-detection/models/cnn_td.h5
gender_model=$proj_dir/models/gender.h5
wav_dir=$expt_dir/wavs
feats_dir=$expt_dir/features
scpfile=$feats_dir/wav.scp
lists_dir=$feats_dir/scp_lists
py_scripts_dir=$proj_dir/python_scripts
#if [ -d "$wav_dir" ]; then rm -rf $wav_dir;fi
mkdir -p $wav_dir $feats_dir/log $lists_dir $expt_dir/SAD/{timestamps,posteriors} $expt_dir/GENDER/{timestamps,posteriors} 

### Create .wav files given movie_files
echo -e "\n >>>> CREATING WAV FILES <<<< "
bash_scripts/create_wav_files.sh $movie_list $wav_dir $nj
num_movies=`cat ${movie_list} | wc -l`
num_wav_extracted=`ls ${wav_dir} | wc -l`
if [ $num_movies -ne $num_wav_extracted ]; then
    echo "Unable to extract all .wav files, exiting..."
#    exit 1
fi

### Extract fbank-features
echo " >>>> EXTRACTING FEATURES FOR SAD <<<< "
bash_scripts/create_logmel_feats.sh $wav_dir $feats_dir $nj
num_feats=`cat ${feats_dir}/feats.scp | wc -l`
if [ $num_movies -ne $num_feats ]; then
    echo "Unable to extract all feature files, exiting..."
#    exit 1
fi

## Generate SAD Labels
echo " >>>> GENERATING SAD LABELS <<<< "
movie_count=1
for movie_path in `cat $movie_list`
do
    movieName=`basename $movie_path | awk -F '.' '{print $1}'`
 #   printf "$movieName\n"
    cat $feats_dir/feats.scp | grep -- "${movieName}" > $lists_dir/${movieName}_feats.scp
    if [ -f $expt_dir/SAD/posteriors/${movieName}.post ];then
        continue
    fi
    python $py_scripts_dir/generate_sad_labels.py -o $sad_overlap --unif_seg $uniform_seg_len $expt_dir $lists_dir/${movieName}_feats.scp $sad_model & 
    if [ $(($movie_count % $nj)) -eq 0 ];then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

if [[ $only_vad -eq 1 ]]; then
    exit 1
fi

### Create VGGish embeddings
echo " >>>> EXTRACTING VGGISH EMBEDDINGS <<<< "
## Download vggish_model.ckpt file if not exists in python_scripts/audioset_scripts/ directory
python $py_scripts_dir/download_vggish_ckpt_file.py python_scripts/audioset_scripts/vggish_model.ckpt
    
ls $wav_dir/*.wav  > $expt_dir/wav.list
python $py_scripts_dir/extract_vggish_feats.py -o $gender_overlap $proj_dir $expt_dir/wav.list $feats_dir/vggish $nj 

#movie_count=1
#for wav_file in `cat $expt_dir/wav.list`
#do
#    movieName=`basename $wav_file .wav`
#    python $py_scripts_dir/extract_vggish_feats.py -o $gender_overlap $proj_dir $wav_file $feats_dir/vggish &
#    if [ $(($movie_count % $nj)) -eq 0 ]; then
#        wait
#    fi
#    movie_count=`expr $movie_count + 1`
#done
#wait

### Make gender predictions
echo " >>>> PREDICTING GENDER SEGMENTS <<<< "
python $py_scripts_dir/predict_gender.py -o $gender_overlap $expt_dir $gender_model $nj

## Delete feature files and/or wav files unless otherwise specified
if [[ "$feats_flag" == "n" ]]; then
    rm -r $feats_dir & 
fi
if [[ "$wavs_flag" == "n" ]]; then
    rm -r $wav_dir &
fi
rm $expt_dir/wav.list 
wait
echo " >>>> GENDER SEGMENTS PER-MOVIE CAN BE FOUND IN $expt_dir/GENDER/timestamps <<<< "

