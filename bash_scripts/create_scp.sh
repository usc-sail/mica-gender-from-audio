##
##      
##      Write kaldi-style wav.scp file for audio files.
##
##      Arguments :
##
##      wav_dir    -    Directory where all split audio segments are stored.
##      scpfile    -    Output wav.scp file    
##                      format :  wav_id <space> path_to_wav_file
##

wav_dir=${1}
scpfile=${2}

find $wav_dir -type f -name '*.wav' | while read r
do 
    echo `basename $r .wav`" "$r 
done > $scpfile

sort $scpfile -o $scpfile

