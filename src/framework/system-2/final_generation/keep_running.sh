#!/bin/bash
until python generate_with_all_evidence.py --dataset "2wikimultihopqa" --batch_size 8; do
    echo "Script crashed with exit code $?. Restarting..." >&2
    sleep 2
done
