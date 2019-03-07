#!/bin/bash
#. ./path.sh

##
##
##
##  Generate gender and VAD timestamps based on models trained on keras. 
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
##      nj            : number of jobs to be run in parallel 
##      feats_flag    : 'y' if you want to retain feature files after execution
##      wavs_flag     : 'y' if you want to retain '.wav' audio files after execution
##      overlap       : % overlap of segments during SAD inference (0-1, i.e, 0 for no overlap, 
##                      1 for complete overlap)
##  
##  Packages/libraries required :
##     kaldi          : ensure that all kaldi binaries are added to system path. If not,
##                      either add them to system path, or modify kaldi_root in 1st line of
##                      'path.sh' to reflect kaldi installation directory.
##     keras, tensorflow              :     required to load model and make VAD predictions.
##     re, gzip, struct               :     used by kaldi_io, to read feature files.
##     Other python libraries required include numpy, scipy, resampy.
##
##
##

## Define usage of script
usage="Perform gender identification from audio
Usage: bash $(basename "$0") [-h] [-w y/n] [-f y/n] [-j num_jobs] [-o overlap] movie_paths.txt (out_dir)
e.g.: bash $(basename "$0") -w y -f y -nj 8 -o 0.5 demo.txt DEMO

where:
-h                  : Show help 
-w                  : Store wav files after processing (default: n)
-f                  : Store feature files after processing (default: n)
-j                 : Number of parallel jobs to run (default: 16)
-o                  : Percentage overlap of segments during SAD (0-1) (default: 0)
movie_paths.txt     : Text file consisting of complete paths to media files (eg, .mp4/.mkv) on each line 
out_dir             : Directory in which to store all output files (default: "\$PWD"/gender_out_dir)
"

## Kill all background processes if file exits due to error
trap "exit" INT TERM
trap "kill 0" EXIT
## Add kaldi binaries to path if path.sh file exists
if [ -f path.sh ]; then . ./path.sh; fi
## Default Options
feats_flag="n"
wavs_flag="n"
nj=16
overlap=0

## Input Options
if [ $# -eq 0 ];
then
    echo "$usage"
    exit
fi

while getopts ":hw:f:j:o:" option
do
    case "${option}"
    in
        h) echo "$usage"
        exit;;
        f) feats_flag="${OPTARG}";;
        w) wavs_flag="${OPTARG}";;
        j) nj=${OPTARG};;
        o) overlap=${OPTARG};;
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

## Reduce nj if not enough files
num_movies=`cat $movie_list | wc -l`
if [ $num_movies -lt $nj ]; then
    nj=$num_movies
fi

proj_dir=$PWD
vad_model=$proj_dir/models/vad.h5
gender_model=$proj_dir/models/gender.h5
wav_dir=$expt_dir/wavs
feats_dir=$expt_dir/features
scpfile=$feats_dir/wav.scp
logfile=$feats_dir/speaker_segmentation.log
lists_dir=$feats_dir/scp_lists
py_scripts_dir=$proj_dir/python_scripts
if [ -d "$wav_dir" ]; then rm -rf $wav_dir;fi
mkdir -p $wav_dir $feats_dir/log $lists_dir $expt_dir/VAD/{spk_seg,timestamps,posteriors} $expt_dir/GENDER/{timestamps,posteriors} 

### Create .wav files given movie_files
echo " >>>> CREATING WAV FILES <<<< "
bash_scripts/create_wav_files.sh $movie_list $wav_dir $nj
num_movies=`cat ${movie_list} | wc -l`
num_wav_extracted=`ls ${wav_dir} | wc -l`
if [ $num_movies -ne $num_wav_extracted ]; then
    echo "Unable to extract all .wav files, exiting..."
    exit 1
fi

### Extract fbank-features
echo " >>>> EXTRACTING FEATURES FOR VAD <<<< "
bash_scripts/create_logmel_feats.sh $wav_dir $feats_dir $nj
num_feats=`cat ${feats_dir}/feats.scp | wc -l`
if [ $num_movies -ne $num_feats ]; then
    echo "Unable to extract all feature files, exiting..."
    exit 1
fi

## Generate VAD Labels
echo " >>>> GENERATING VAD LABELS <<<< "
movie_count=1
for movie_path in `cat $movie_list`
do
    movieName=`basename $movie_path | awk -F '.' '{print $1}'`
    cat $feats_dir/feats.scp | grep -- "${movieName}" > $lists_dir/${movieName}_feats.scp
    python $py_scripts_dir/generate_vad_labels.py $expt_dir $lists_dir/${movieName}_feats.scp $vad_model $overlap & 
    if [ $(($movie_count % $nj)) -eq 0 ];then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

### Create VGGish embeddings
echo " >>>> SPEAKER SEGMENTATION / GENERATE VGGISH EMBEDDINGS <<<< "
## Download vggish_model.ckpt file if not exists in python_scripts/audioset_scripts/ directory
python $py_scripts_dir/download_vggish_ckpt_file.py python_scripts/audioset_scripts/vggish_model.ckpt

ls $wav_dir/*.wav  > $expt_dir/wav.list
movie_count=1
for wav_file in `cat $expt_dir/wav.list`
do
    movieName=`basename $wav_file .wav`
    extract-segments scp:$scpfile $expt_dir/VAD/timestamps/${movieName}_wo_ss.ts ark:- 2>>$logfile | \
     compute-mfcc-feats ark:- ark:- 2>>$logfile | \
      spk-seg --bic-alpha=1.1 ark:- $expt_dir/VAD/spk_seg/$movieName.seg 2>>$logfile
    python $py_scripts_dir/spk_seg_to_vad_ts.py $movieName $expt_dir & 
    python $py_scripts_dir/compute_and_write_vggish_feats.py $proj_dir $wav_file $feats_dir/vggish &
    if [ $(($movie_count % $nj)) -eq 0 ]; then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

### Make gender predictions
echo " >>>> PREDICTING GENDER SEGMENTS <<<< "
python $py_scripts_dir/predict_gender.py $expt_dir $gender_model

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

