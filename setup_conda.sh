#!/bin/zsh

conda create --prefix twibot python=3.12 -y
conda activate twibot/
pip install -r requirements.txt