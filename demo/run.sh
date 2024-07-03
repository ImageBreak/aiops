#!/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl