#!/bin/bash

# Argument indexes
length=$(($#-1))
input=${@:1:$length}
output=${@:$#}

# Verbose arguments
echo -e "\u001b[36;1mInput directories:\u001b[0m"
echo $input
echo -e "\u001b[36;1m\nOutput directories:\u001b[0m"
echo $output
echo -e "\u001b[36;1m\nLog:\u001b[0m"

# Create output directories
rm -rf $output
for directory in $input
do
    echo "dir: $directory"
    for subdirectory in ${directory}/*
    do
        bn=$(basename -- $subdirectory)
        mkdir -vp "$output/train/$bn"
        mkdir -vp "$output/test/$bn"
        mkdir -vp "$output/validation/$bn"
    done
done

#Fill them with images
for directory in $input
do
    echo "dir: $directory"
    for subdirectory in ${directory}/*
    do
        for file in ${subdirectory}/*
        do
            rand1000=$((1 + $RANDOM % 1000 ))
            if [[ $rand1000 -lt 600 ]]
            then
                cp $file $output/train/$(basename -- $subdirectory)
            elif [ $rand1000 -lt 800 ]
            then
                cp $file $output/test/$(basename -- $subdirectory)
            else
                cp $file $output/validation/$(basename -- $subdirectory)
            fi
        done
    done
done
