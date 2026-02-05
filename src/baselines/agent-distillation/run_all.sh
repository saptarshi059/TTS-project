#!/bin/bash
#This tells the shell to stop immediately if any command returns an error.
#Without it, if the first script crashes 5 minutes after you go to bed, the second one will still run.
set -e

python ircot_frames.py
python ircot_three_ds.py