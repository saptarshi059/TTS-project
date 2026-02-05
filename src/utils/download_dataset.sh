#!/bin/bash

dataset_name=$1
output_dir="../../data/$dataset_name"

if [ "$dataset_name" = '2wikimultihopqa' ]; then
  mkdir -p "$output_dir"
  gdown --id '13WXwB1yC-bUJgrQdAOSgmUUUJTS8qns3' -O "$output_dir/"
  gdown --id '13WPx6-JhAhh9LqQBLep5VvRWXALlxKDs' -O "$output_dir/"
fi

if [ "$dataset_name" = 'musique' ]; then
  mkdir -p "$output_dir"
  gdown --id '14EK5bYcQKbH28CmIEPzlt08c3B3h1gKb' -O "$output_dir/"
  gdown --id '14ItAhu2AmKzJBsI4qZW_OSVslApFS4nR' -O "$output_dir/"
fi

if [ "$dataset_name" = 'hotpotqa' ]; then
  mkdir -p "$output_dir"
  gdown --id '140tEaZXCpiBSsa8ET_tm7MDiD1voTTXR' -O "$output_dir/"
  gdown --id '13o0E0qfnk3QhGOKgJ9SegDRBGB_tf_uE' -O "$output_dir/"
fi