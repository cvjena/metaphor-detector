#!/bin/sh
mkdir -p sentences
counter=0
num_files=`ls /lvdata/compling/gerdracor/processed_sch | wc -l`
for fullfile in /lvdata/compling/gerdracor/processed_sch/*; do
    i=$((i+1))
    echo "file $i of $num_files"
    filename=`echo $fullfile | sed 's:.*/::'`
    vectorfile="sentences_sch/$filename"
    python get_sentence_boundaries_sch.py $fullfile $vectorfile
done
echo "done"
