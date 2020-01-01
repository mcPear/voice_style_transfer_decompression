#!/bin/bash
b=8
dir=wav48
compr_dir=$dir-"$b"-bit
mkdir $compr_dir
for d in ./$dir/*/ ; do
    	echo "$d"
	speaker_dir=$(basename "$d")
	compr_speaker_dir=./"$compr_dir"/"$speaker_dir"
	mkdir $compr_speaker_dir
	for filename in "$d"*; do
		echo "$filename"
		base_file=$(basename "$filename" .wav)
		echo "$base_file"
		lame -b "$b" "$filename" "$compr_speaker_dir/$base_file.mp3"
	done
done
