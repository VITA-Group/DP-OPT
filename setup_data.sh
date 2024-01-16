#!/bin/bash

# Create data folder
mkdir -p data

# Get Ordered Prompt data
wget https://github.com/yaolu/Ordered-Prompt/archive/refs/heads/main.zip
unzip main.zip
mv Ordered-Prompt-main/data data/ordered_prompt
rm -rf Ordered-Prompt-main
rm -f main.zip
