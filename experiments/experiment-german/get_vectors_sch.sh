#!/bin/sh

counter=0
num_files=`ls /lvdata/compling/gerdracor/processed_sch | wc -l`
for fullfile in /lvdata/compling/gerdracor/processed_sch/*; do
    i=$((i+1))
    echo "file $i of $num_files"
    filename=`echo $fullfile | sed 's:.*/::'`
    vectorfile="vectors_sch/$filename"
    python get_vectors_sch.py $fullfile $vectorfile
done
echo "done"
