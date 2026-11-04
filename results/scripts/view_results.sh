#!/bin/bash
# view_results.sh — Summary of all experiments
RESULTS_DIR="/data/results/hangar/rosbags_1"
printf "%-32s %10s %10s %10s\n" "EXPERIMENT" "PCD" "SIZE" "TRAJ"
printf "%-32s %10s %10s %10s\n" "----------" "---" "----" "----"
for d in "$RESULTS_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    n_pcd=$(ls "$d/Log/PCD"/*.pcd 2>/dev/null | wc -l)
    size=$(du -sh "$d/Log/PCD" 2>/dev/null | cut -f1)
    if [ -f "$d/Log/mat_out.txt" ]; then
        traj=$(wc -l < "$d/Log/mat_out.txt")
    else
        traj=0
    fi
    printf "%-32s %10s %10s %10s\n" "$name" "$n_pcd" "$size" "$traj"
done
