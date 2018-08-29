##
##      
##      Perform speaker-homogenous segmentation of audio
##
##      Arguments :
##
##      wav_dir    -    Directory where speech segments (post-VAD) 
##                      are stored as audio .wav files
##      scpfile    -    Output wav.scp file    
##                      format :  wav_id <space> path_to_wav_file
##

movie_list=${1}
wav_dir=${2}
write_dir=${3}

scpfile=$wav_dir/all.scp
find $wav_dir -type f -name '*.wav' | while read r
do 
    echo `basename $r .wav`" "$r 
done > $scpfile
sort $scpfile -o $scpfile

for movie_path in `cat $movie_list`
do
    movie=`basename $movie_path | awk -F '.' '{print $1}'`
    cat $scpfile | grep -- $movie > $wav_dir/$movie.scp
    compute-mfcc-feats scp:$wav_dir/$movie.scp ark:$write_dir/tmp.ark 2>/dev/null
    spk-seg --bic-alpha=1.1 ark:$write_dir/tmp.ark $write_dir/$movie.seg 2>/dev/null
    rm $write_dir/tmp.ark
done

