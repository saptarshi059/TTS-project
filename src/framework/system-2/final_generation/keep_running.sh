#!/bin/bash
until python generate_with_all_evidence.py --dataset "frames" --batch_size 4; do
    echo "Script crashed with exit code $?. Restarting..." >&2
    sleep 2
done
