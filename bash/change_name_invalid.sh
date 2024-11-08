#!/bin/bash

# Set the directory where the files are located
dir_path="/home/francobertoldi/Documents/RelationNetworkDNI/data/train/invalid"

# Loop through the files in the directory
i=1
for file in "$dir_path"/*
do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]
    then
        # Construct the new filename
        new_filename="invalid_image_${i}.jpg"
        
        # Rename the file
        mv "$file" "$dir_path/$new_filename"
        
        # Increment the counter
        ((i++))
    fi
done
