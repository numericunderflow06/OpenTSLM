#!/bin/bash
#
# Download ZuCo 2.0 TSR (Task-Specific Reading) Matlab result files from OSF
# These are needed for the ZuCo 2.0 ET reading task classification experiments.
#
# Files are large (1-3 GB each, ~30 GB total). Ensure enough disk space.
#

set -e

DEST_DIR="/home/wangni/tslm-co/data/zuco/raw/zuco2/2urht/osfstorage/task2 - TSR/Matlab files"

mkdir -p "$DEST_DIR"
echo "Downloading ZuCo 2.0 TSR Matlab files to: $DEST_DIR"
echo ""

# OSF download URLs for all 18 TSR result files
declare -A FILES=(
    ["resultsYAC_TSR.mat"]="https://osf.io/download/923qd/"
    ["resultsYAG_TSR.mat"]="https://osf.io/download/u9rz5/"
    ["resultsYAK_TSR.mat"]="https://osf.io/download/w3rx6/"
    ["resultsYDG_TSR.mat"]="https://osf.io/download/4ecyr/"
    ["resultsYDR_TSR.mat"]="https://osf.io/download/wbnfm/"
    ["resultsYFR_TSR.mat"]="https://osf.io/download/xr9km/"
    ["resultsYFS_TSR.mat"]="https://osf.io/download/tc8en/"
    ["resultsYHS_TSR.mat"]="https://osf.io/download/4qw36/"
    ["resultsYIS_TSR.mat"]="https://osf.io/download/cvtdk/"
    ["resultsYLS_TSR.mat"]="https://osf.io/download/8rs5f/"
    ["resultsYMD_TSR.mat"]="https://osf.io/download/hw8yf/"
    ["resultsYMS_TSR.mat"]="https://osf.io/download/uaq3h/"
    ["resultsYRH_TSR.mat"]="https://osf.io/download/cg4jq/"
    ["resultsYRK_TSR.mat"]="https://osf.io/download/jcwfb/"
    ["resultsYRP_TSR.mat"]="https://osf.io/download/gxpqy/"
    ["resultsYSD_TSR.mat"]="https://osf.io/download/devqp/"
    ["resultsYSL_TSR.mat"]="https://osf.io/download/3ba2g/"
    ["resultsYTL_TSR.mat"]="https://osf.io/download/5wq4g/"
)

TOTAL=${#FILES[@]}
COUNT=0

for filename in $(echo "${!FILES[@]}" | tr ' ' '\n' | sort); do
    url="${FILES[$filename]}"
    dest_path="$DEST_DIR/$filename"
    COUNT=$((COUNT + 1))

    if [ -f "$dest_path" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $filename"
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading: $filename"
    curl -L -o "$dest_path" "$url" --progress-bar
    echo "  Done: $(du -h "$dest_path" | cut -f1)"
done

echo ""
echo "Download complete. Files in: $DEST_DIR"
ls -lh "$DEST_DIR"
