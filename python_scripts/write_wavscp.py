import sys

wav_list = sys.argv[1]
write_dir = sys.argv[-1]
write_scp = write_dir + 'wav.scp'

wavs = [x.rstrip() for x in open(wav_list,'r').readlines()]
fw = open(write_scp,'w')
#fw_utt = open(feats_dir + 'utt2spk','w')
for wav_path in wavs:
    seg_name = wav_path[wav_path.rfind('/')+1:wav_path.rfind('.wav')]
    fw.write(seg_name + ' ' + wav_path + '\n')
#        fw_utt.write(seg_files[:-4]+' '+seg_files[:-4]+'\n')
fw.close()
