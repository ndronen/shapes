#!/bin/bash

dataset_dir=$1
shift
if [ $# -eq 1 ]
then
    max=$1
    shift
else
    max=99999999
fi

if [ ! -d "$dataset_dir" ]
then
    echo "usage: $(basename $0) DATASET_DIR" >&2
    exit 1
fi

if [ ! -d "$dataset_dir"/images ]
then
    echo "usage: $(basename $0) DATASET_DIR" >&2
    echo "DATASET_DIR must contain 'images' and 'labels' subdirectories" >&2
    exit 1
fi

i=1
for image in $(find $dataset_dir/images/ -name '*.jpg')
do
    label=$(echo $image | sed -e 's,image,label,' -e 's,jpg,png,')
    bin/plot $image $label
    i=$(($i+1))
    if [ $i -gt $max ]
    then
        break
    fi
done
