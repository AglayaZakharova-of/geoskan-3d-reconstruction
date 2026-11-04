#!/bin/bash
# run_exp.sh — Run one FAST-LIVO2 experiment
# Usage: ./run_exp.sh <exp_name> [filter_surf=0.1] [filter_pcd=0.05] [interval=100] [bag=bag_1_humble]

set -e

EXP_NAME="${1:?Usage: ./run_exp.sh <exp_name> [filter_surf] [filter_pcd] [interval] [bag]}"
FILTER_SURF="${2:-0.1}"
FILTER_PCD="${3:-0.05}"
INTERVAL="${4:-100}"
BAG_NAME="${5:-bag_1_humble}"

BAG_PATH="/data/source/hangar/rosbags_1/${BAG_NAME}"
CONFIG="$HOME/ros2_ws/src/FAST-LIVO2/config/MARS_LVIG.yaml"
BASELINE_CONFIG="/data/results/scripts/MARS_LVIG_baseline.yaml"
LOG_DIR="$HOME/ros2_ws/src/FAST-LIVO2/Log"
OUT_DIR="/data/results/hangar/rosbags_1/${EXP_NAME}"

[ -d "$BAG_PATH" ]        || { echo "ERROR: bag not found: $BAG_PATH" >&2; exit 1; }
[ -f "$BASELINE_CONFIG" ] || { echo "ERROR: baseline config not found" >&2; exit 1; }

set +u
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
set -u

echo "[0/6] Restoring baseline config"
cp "$BASELINE_CONFIG" "$CONFIG"

echo "[1/6] filter_surf=$FILTER_SURF filter_pcd=$FILTER_PCD interval=$INTERVAL"
sed -i "s/^\([[:space:]]*\)filter_size_surf: .*/\1filter_size_surf: ${FILTER_SURF}/" "$CONFIG"
sed -i "s/^\([[:space:]]*\)filter_size_pcd: .*/\1filter_size_pcd: ${FILTER_PCD}/" "$CONFIG"
sed -i "s/^\([[:space:]]*\)interval: .*/\1interval: ${INTERVAL}/" "$CONFIG"

grep -E "filter_size_surf|filter_size_pcd|^[[:space:]]*interval:" "$CONFIG"

echo "[2/6] Cleaning previous results"
mkdir -p "$LOG_DIR/PCD"
rm -f "$LOG_DIR/PCD"/*.pcd
rm -f "$LOG_DIR"/mat_out.txt "$LOG_DIR"/imu.txt "$LOG_DIR"/mat_pre.txt

echo "[3/6] Starting ros2 bag play (background)"
ros2 bag play "$BAG_PATH" > "/tmp/play_${EXP_NAME}.log" 2>&1 &
PLAY_PID=$!

sleep 3

echo "[4/6] Starting FAST-LIVO2 (background)"
cd "$HOME/ros2_ws/src/FAST-LIVO2"
ros2 launch fast_livo mapping_avia_marslvig.launch.py > "/tmp/launch_${EXP_NAME}.log" 2>&1 &
LAUNCH_PID=$!

echo "[5/6] Waiting for bag to finish..."
wait "$PLAY_PID" 2>/dev/null || true
echo "      bag finished, waiting 5s for last PCD writes..."
sleep 5

kill -INT  "$LAUNCH_PID" 2>/dev/null || true
sleep 3
kill -KILL "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true

echo "[6/6] Copying results to $OUT_DIR"
mkdir -p "$OUT_DIR/Log/PCD"
cp "$LOG_DIR/PCD"/*.pcd "$OUT_DIR/Log/PCD/" 2>/dev/null || true
for f in mat_out.txt imu.txt mat_pre.txt scans_pos.json; do
    [ -f "$LOG_DIR/$f" ] && cp "$LOG_DIR/$f" "$OUT_DIR/Log/"
done
cp "$CONFIG" "$OUT_DIR/config_used.yaml"
cp "/tmp/play_${EXP_NAME}.log"   "$OUT_DIR/play.log"   2>/dev/null || true
cp "/tmp/launch_${EXP_NAME}.log" "$OUT_DIR/launch.log" 2>/dev/null || true

N_PCD=$(ls "$OUT_DIR/Log/PCD"/*.pcd 2>/dev/null | wc -l)
SIZE=$(du -sh "$OUT_DIR/Log/PCD" 2>/dev/null | cut -f1)
echo "================================================================"
echo " Experiment:      $EXP_NAME"
echo " filter_size_surf: $FILTER_SURF"
echo " filter_size_pcd:  $FILTER_PCD"
echo " interval:         $INTERVAL"
echo " PCD files:        $N_PCD"
echo " PCD total size:   $SIZE"
echo " Results:          $OUT_DIR"
echo "================================================================"
