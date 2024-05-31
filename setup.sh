#!/bin/bash

micromamba activate lhe
FW_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$FW_PATH:$PYTHONPATH
