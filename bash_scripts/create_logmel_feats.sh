#!/bin/bash

## 
##  Author/year: Rajat Hebbar, 2019
##
##  Extract 64D log-Mel filterbank energy features
##
##  Input 
##      wav_dir     - Directory in which .wav files are stored
##      feats_dir   - Directory in which to store features
##      nj          - Number of parallel jobs to run
##
##  Usage
##      bash create_logmel_feats.sh [wav_dir] [feats_dir] [nj]
##
##  Example
##      bash create_logmel_feats.sh gender_out_dir/wavs gender_out_dir/features 4  
##

wav_dir=$1
feats_dir=$2
nj=$3

log_dir=$feats_dir/log
fbank_dir=$feats_dir/fbank_data
scp=$feats_dir/wav.scp
mkdir -p $log_dir $fbank_dir 
if [ -f path.sh ];  then ./path.sh; fi 

## Create 'wav.scp' file for kaldi feature extraction

find $wav_dir -type f -name '*.wav' | while read r
do
    movie_name=`basename $r .wav`
    echo $movie_name" "$r
done > $scp

sort $scp -o $scp

####
####    Extract log-Mel filterbank coefficients
####

## Split wav.scp into 'nj' parts
split_wav_scp=""
for n in $(seq $nj); do
    split_wav_scp="$split_wav_scp $log_dir/wav.scp.$n"
done
./split_scp.pl $scp $split_wav_scp || exit 1;

## Extract fbank features using run.pl parallelization
./run.pl JOB=1:$nj $log_dir/make_fbank_feats.JOB.log \
    compute-fbank-feats --verbose=2 --num-mel-bins=64 scp:$log_dir/wav.scp.JOB ark,p:- \| \
    copy-feats --compress=true ark,p:- \
        ark,scp,p:$fbank_dir/raw_fbank_feats.JOB.ark,$fbank_dir/raw_fbank_feats.JOB.scp \
|| exit 1;

## Combine multiple fbank files 
for n in $(seq $nj); do
  cat $fbank_dir/raw_fbank_feats.$n.scp || exit 1;
done > $feats_dir/feats.scp

