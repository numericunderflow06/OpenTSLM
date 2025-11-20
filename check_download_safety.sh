#!/bin/bash
# Safety check script to ensure no data goes to /home/wangni

echo "==================================================================="
echo "Download Path Safety Check"
echo "==================================================================="
echo ""

# Check if any paths point to /home/wangni
check_path() {
    local path=$1
    local name=$2
    
    if [[ "$path" == /home/wangni* ]]; then
        echo "❌ ERROR: $name points to /home/wangni"
        echo "   Path: $path"
        echo "   This will cause cluster storage quota issues!"
        return 1
    elif [[ "$path" == /local/home/wangni* ]]; then
        echo "✅ SAFE: $name points to /local/home/wangni"
        echo "   Path: $path"
        return 0
    else
        echo "⚠️  WARNING: $name points to unexpected location"
        echo "   Path: $path"
        return 1
    fi
}

# Check current directory
echo "1. Current working directory:"
check_path "$(pwd)" "PWD"
echo ""

# Check data directory
echo "2. OpenTSLM data directory:"
DATA_DIR="/local/home/wangni/OpenTSLM/data"
check_path "$DATA_DIR" "DATA_DIR"
echo ""

# Check for free space
echo "3. Disk space check:"
df -h /local/home/wangni | head -2
echo ""

# Summary
echo "==================================================================="
echo "Summary:"
echo "  - All paths should point to /local/home/wangni"
echo "  - NEVER use /home/wangni (will cause cluster ban)"
echo "  - Minimum 100 GB free space recommended for MIMIC-IV-ECG"
echo "==================================================================="
