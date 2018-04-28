##
##
##      Extract audio (mono,sampled at 8kHz) from given AV files
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

wav_dir=$expt_dir/wavs             ## Directory in which to store complete audio files

movie_num=1
for mov_file in `cat ${movie_paths}`
do
    base=`basename $mov_file`
    movie_name=`echo $base | awk -F '.' '{ print $1 }'` 

    ffmpeg -loglevel panic -i ${mov_file} -ar 8k -ac 1 ${wav_dir}/${movie_name}.wav &  ## Extract single-channel audio from input sampled at 8000 Hz.
    if [ $(($movie_num % $nj )) -eq 0 ]
    then
        wait
    fi
    movie_num=`expr $movie_num + 1`
done
wait
