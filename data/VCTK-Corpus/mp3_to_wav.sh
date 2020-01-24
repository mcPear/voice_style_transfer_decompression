#!/bin/bash
b=wav
dir=wav48-8
compr_dir=$dir-"$b"
mkdir $compr_dir
for d in ./$dir/*/ ; do
    	echo "$d"
	speaker_dir=$(basename "$d")
	compr_speaker_dir=./"$compr_dir"/"$speaker_dir"
	mkdir $compr_speaker_dir
	for filename in "$d"*; do
		echo "$filename"
		base_file=$(basename "$filename" .mp3)
		echo "$base_file"
		lame --decode "$filename" "$compr_speaker_dir/$base_file.wav"
	done
done
