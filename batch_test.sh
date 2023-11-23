#!/bin/bash
filename="$HOME/myubai/AutocompleteLM/test.ipynb"

sbatch "sbatch_script.sbatch" "$filename"
