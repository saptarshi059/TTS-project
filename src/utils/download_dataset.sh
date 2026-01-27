#!/bin/bash

dataset_name=$1
output_dir="../data/$dataset_name"

if [ "$dataset_name" = '2wiki' ]; then
  mkdir -p "$output_dir"
  gdown -id '13WXwB1yC-bUJgrQdAOSgmUUUJTS8qns3' -O "$output_dir/"
  gdown -id '13WPx6-JhAhh9LqQBLep5VvRWXALlxKDs' -O "$output_dir/"
fi