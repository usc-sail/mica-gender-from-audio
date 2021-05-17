nj=64                # Number of files to process simultaneously
feats_flag="n"       # "y"/"n" flag to keep kaldi-feature files after inference
wavs_flag="y"        # "y"/"n" flag to keep .wav audio files after inference
sad_overlap=0.0        # % overlap in SAD-segments (range: 0-1, 0 for no overlap) (single segment is 0.64s)
gender_overlap=0     # % overlap in GENDER-segment 
uniform_seg_len=1    # Segment length for uniform speaker-segmentation 
only_vad=0           # 1 if only vad else 0
