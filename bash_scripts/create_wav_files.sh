#!/bin/bash

##
##  Author/year: Rajat Hebbar, 2019
##
##  Extract audio (mono,sampled at 16kHz) from given AV files
##
##  Input
##      movie_paths :  List of paths to input files to process
##      wav_dir     :  Directory in which to store audio files 
##      nj          :  Number of parallel jobs to process
##
##  Usage
##      bash create_wav_files.sh [movie_paths] [wav_dir] [nj]
##  
##  Example
##      bash create_wav_files.sh movie_files.txt gender_out_dir/wavs 4
##


movie_paths=${1}
wav_dir=${2}
nj=${3}


movie_num=1
for mov_file in `cat ${movie_paths}`
do
    base=`basename -- $mov_file`
    movie_name=`echo $base | awk -F '.' '{ print $1 }'` 

    ffmpeg -loglevel error -i ${mov_file} -ar 16k -ac 1 ${wav_dir}/${movie_name}.wav & ## Extract single-channel audio from input sampled at 16000 Hz.
    if [ $(($movie_num % $nj )) -eq 0 ]
    then
        wait
    fi
    movie_num=`expr $movie_num + 1`
done
wait

for mov_file in `cat ${movie_paths}`
do
    base=`basename -- $mov_file`
    movie_name=`echo $base | awk -F '.' '{ print $1 }'` 
    if [ ! -e ${wav_dir}/${movie_name}.wav ]; then
        echo "Unable to extract ${mov_file}, please check if corrupted file"
    fi
done
