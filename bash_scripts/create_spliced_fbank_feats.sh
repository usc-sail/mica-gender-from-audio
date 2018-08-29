## 
##
##  Extract spliced log-Mel filterbank energy features
##
##  Arguments :
##
##      wav_dir     - Directory in which sampled .wav files are 
##                      stored
##      feats_dir   - Directory in which to store all features
##      nj          - Number of parallel jobs to run
##

wav_dir=$1
feats_dir=$2
nj=$3

log_dir=$feats_dir/log
fbank_dir=$feats_dir/fbank_data
spl_dir=$feats_dir/spliced_fbank_data
segments=$feats_dir/segments
scp=$feats_dir/wav.scp

mkdir -p $log_dir $fbank_dir $spl_dir

if [ -f path.sh ];  then ./path.sh; fi 

## Create 'segments' and 'wav.scp' files for kaldi feature extraction
if [ -f $segments ]; then rm $segments; fi

find $wav_dir -type f -name '*.wav' | while read r
do
    movie_name=`basename $r .wav`
    movie_time=`soxi -D ${wav_dir}/${movie_name}.wav`
    movie_time_int=`echo ${movie_time} | awk -F '.' '{ print $1 }'`
    
    ## Create segments file for kaldi feature extraction by partitioning each wav file into 1sec segments
    for n in `seq $movie_time_int`
    do
        segnum=`printf "%05d" $n`
        echo "${movie_name}_seg-${segnum} $movie_name $((n-1)) ${n}.015"     ## Extra 0.015sec required due to overlapping windows in feature-extraction
    done >> $segments
    ## Final segment
    segnum=`printf "%05d" $((n+1))`
    echo "${movie_name}_seg-${segnum} $movie_name $n $movie_time" >> $segments
    echo $movie_name" "$r
done > $scp

sort $scp -o $scp
sort $segments -o $segments

####
####    Extract log-Mel filterbank coefficients
####

## Split wav.scp into 'nj' parts
split_segments=""
for n in $(seq $nj); do
    split_segments="$split_segments $log_dir/segments.$n"
done
./split_scp.pl $segments $split_segments || exit 1;

## Extract fbank features using run.pl parallelization
./run.pl JOB=1:$nj $log_dir/make_fbank_feats.JOB.log \
    extract-segments scp,p:$scp $log_dir/segments.JOB ark:- \| \
    compute-fbank-feats --verbose=2 --sample-frequency=8000 ark,p:- ark,p:- \| \
    copy-feats --compress=true ark,p:- \
        ark,scp,p:$fbank_dir/raw_fbank_feats.JOB.ark,$fbank_dir/raw_fbank_feats.JOB.scp \
# || exit 1;

## Combine multiple fbank files and delete segment split files
rm $log_dir/segments* 2>/dev/null 
for n in $(seq $nj); do
  cat $fbank_dir/raw_fbank_feats.$n.scp || exit 1;
done > $feats_dir/feats.scp
 
### 
###     Apply mean and variance normalization to filterbank features.
###
compute-cmvn-stats scp:$feats_dir/feats.scp ark,scp,t:$feats_dir/cmvn_stats.ark,$feats_dir/cmvn_stats.scp 2>/dev/null
apply-cmvn --norm-vars=true scp:$feats_dir/cmvn_stats.scp scp,p:$feats_dir/feats.scp ark,scp,t:$feats_dir/norm_feats.ark,$feats_dir/norm_feats.scp 2>/dev/null

###
###     Splice features with +/- 15 frames context
###

## Split feats.scp into 'nj' parts
mkdir $feats_dir/split
for (( n=0; n<$nj; n++ )); do
    file_id=$(( n + 1 ))
    ./split_scp.pl -j $nj $n $feats_dir/norm_feats.scp $feats_dir/split/norm_feats.$file_id.scp
done

## Splice fbank features 
./run.pl JOB=1:$nj $log_dir/splice_mfcc_feats.JOB.log \
  splice-feats --left-context=15 --right-context=0 \
   scp:$feats_dir/split/norm_feats.JOB.scp \
   ark,scp,t:$fbank_dir/splice_mfcc_feats.JOB.ark,$fbank_dir/splice_mfcc_feats.JOB.scp \
    || exit 1;

# concatenate the spliced feature .scp files together.
for n in $(seq $nj); do
  cat $fbank_dir/splice_mfcc_feats.$n.scp || exit 1;
done > $feats_dir/spliced_feats.scp || exit 1

rm -r $feats_dir/split
