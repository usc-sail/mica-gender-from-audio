import os, sys

movie, expt_dir = sys.argv[1:]
VAD_ts_file = os.path.join(expt_dir, 'VAD/timestamps/', movie + '_wo_ss.ts')
spk_seg_file = os.path.join(expt_dir, 'VAD/spk_seg/', movie + '.seg')

spk_seg_data = [x.rstrip().split() for x in open(spk_seg_file,'r').readlines()]
spk_seg_times = [[x[1], float(x[2]), float(x[3])] for x in spk_seg_data]
vad_data = [x.rstrip().split() for x in open(VAD_ts_file,'r').readlines()]
vad_times = [[float(x[0]), float(x[1])] for x in vad_data]
fw = open(os.path.join(expt_dir, 'VAD/timestamps/', movie + '.ts'),'w')

#print(movie, len(spk_seg_times), len(vad_times))
if len(spk_seg_times)!=0:
    for seg in spk_seg_times:
        try:
            vad_num = int(seg[0].split('_vad-')[1])-1
            start = vad_times[vad_num][0] + seg[1]
            end = vad_times[vad_num][0] + seg[2]
            fw.write('{} {}\n'.format(start, end))
        except:
            continue
fw.close()
    
    
