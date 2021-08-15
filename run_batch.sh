#!/bin/bash

for filename in "${1}"/*
do
  echo "python main.py $filename"
  python main.py --file "$filename"
  echo ""
done
