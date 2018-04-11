##
##
##      Extract audio (mono,sampled at 8kHz) from given AV files
##      Split into 1sec long wav segments, for the purpose of feature
##      normalization.
##
##
##      Arguments :
##      movie_paths :  List of paths to input files to process
##      expt_dir    :  Directory in which to store all output files 
##      nj          :  Number of parallel jobs to process
##
##


movie_paths=${1}
expt_dir=${2}
nj=${3}

full_wav_dir=$expt_dir/wavs             ## Directory in which to store complete audio files
wav_dir=$expt_dir/wav_segments          ## Directory in which to store split wav segments
seg_time=1.015                          ## 1 second long segments (+0.015 for feature extraction)

movie_num=1
for mov_file in `cat ${movie_paths}`
do
    base=`basename $mov_file`
    movieName=`echo $base | awk -F '.' '{ print $1 }'` 
    ffmpeg -i ${mov_file} -ar 8k -ac 1 ${full_wav_dir}/${movieName}.wav &  ## Extract single-channel audio from input sampled at 8000 Hz.
    if [ $(($movie_num % $nj )) -eq 0 ]
    then
        wait
    fi
    movie_num=`expr $movie_num + 1`
done

wait

movie_num=1
for mov_file in `cat ${movie_paths}`
do
    base=`basename $mov_file`
    movieName=`echo $base | awk -F '.' '{ print $1 }'`
    movie_time=`soxi -D ${full_wav_dir}/${movieName}.wav | awk -F '.' '{ print $1 }'`
    mkdir -p $wav_dir/$movieName
    (
    ## Split complete wav files into 1second long wav files.
    for (( ind=0; ind<$movie_time; ind++ ))
    do
        segnum=`printf "%05d" $((ind+1))`
        sox -V1 ${full_wav_dir}/${movieName}.wav $wav_dir/$movieName/${movieName}_seg-${segnum}.wav trim $ind $seg_time
    done
    ) &
    
    if [ $(($movie_num % $nj )) -eq 0 ]
    then
        wait
    fi
    movie_num=`expr $movie_num + 1`    
done
wait
