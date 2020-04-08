#! /usr/bin/env bash

for f in *.py
do
    echo "running example $f"
    python $f
done
