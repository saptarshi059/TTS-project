#!/bin/bash
until python generate_with_all_evidence.py --dataset "frames"; do
    echo "Script crashed with exit code $?. Restarting..." >&2
    sleep 2
done
