## 
##
##  Extract spliced log-Mel filterbank energy features
##  given a kaldi-style wav.scp file in feats_dir directory.
##
##  Arguments :
##
##      feats_dir   - Directory in which input wav.scp exits, and
##                      to store all features
##      nj          - Number of parallel jobs to run
##

feats_dir=$1
nj=$2

log_dir=$feats_dir/log
fbank_dir=$feats_dir/fbank_data
spl_dir=$feats_dir/spliced_fbank_data
scp=$feats_dir/wav.scp

mkdir -p $log_dir $fbank_dir $spl_dir

if [ -f path.sh ];  then ./path.sh; fi 

####
####    Extract log-Mel filterbank coefficients
####

## Split wav.scp into 'nj' parts
split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $log_dir/wav.$n.scp"
done
./split_scp.pl $scp $split_scps || exit 1;

## Extract fbank features using run.pl parallelization
./run.pl JOB=1:$nj $log_dir/make_fbank_feats.JOB.log \
compute-fbank-feats --verbose=2 --sample-frequency=8000 scp,p:$log_dir/wav.JOB.scp ark:- \| \
copy-feats --compress=true ark:- \
 ark,scp:$fbank_dir/raw_fbank_feats.JOB.ark,$fbank_dir/raw_fbank_feats.JOB.scp \
 || exit 1;

## Combine multiple fbank files
for n in $(seq $nj); do
  cat $fbank_dir/raw_fbank_feats.$n.scp || exit 1;
done > $feats_dir/feats.scp
 
### 
###     Apply mean and variance normalization to filterbank features.
###
compute-cmvn-stats scp:$feats_dir/feats.scp ark,scp,t:$feats_dir/cmvn_stats.ark,$feats_dir/cmvn_stats.scp  
apply-cmvn --norm-vars=true scp:$feats_dir/cmvn_stats.scp scp,p:$feats_dir/feats.scp ark,scp,t:$feats_dir/norm_feats.ark,$feats_dir/norm_feats.scp 

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
