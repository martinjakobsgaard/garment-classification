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

# Remove and create output directory
rm -rf $output
mkdir -v $output
touch $output/labels.csv

# Copy files and generate labels
for directory in $input
do
    echo "dir: $directory"
    for file in ${directory}/*
    do
        cp -v $file $output
        echo "$(basename -- $file),$(basename -- $directory)" >> $output/labels.csv
    done
done
