#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_PATH="experiments/scripts/coevolution/run_coevolution.py"
START_GLOBAL=0
END_GLOBAL=20

# ------------------------------------------
# 🎛️ SCALING KNOB
# ------------------------------------------
TOTAL_PARALLEL_STREAMS=3 

# ==========================================
# 🛑 ARCHITECTURAL FIX 1: JOB CONTROL
# ==========================================
# 'set -m' enables Job Control. 
# Implication: Background jobs (&) get their own Process Group ID (PGID).
# Benefit: We can use 'kill -9 -PGID' to wipe the tree, without 'setsid'.
set -m

# ==========================================
# 🛑 ARCHITECTURAL FIX 2: RESOURCE MATH
# ==========================================
# We must limit how many workers each stream uses, or they will kill the CPU.

# 1. Get Total Physical Cores
TOTAL_CORES=$(nproc)

# 2. Calculate Cores Per Stream
# Subtract 2 for system overhead/OS
AVAILABLE_CORES=$((TOTAL_CORES - 2))
WORKERS_PER_STREAM=$((AVAILABLE_CORES / TOTAL_PARALLEL_STREAMS))

# Ensure at least 1 worker
if [ "$WORKERS_PER_STREAM" -lt 1 ]; then
    WORKERS_PER_STREAM=1
fi

echo "🧮 Resource Partitioning Config:"
echo "   Total Cores: $TOTAL_CORES"
echo "   Streams: $TOTAL_PARALLEL_STREAMS"
echo "   Workers allowed per Stream: $WORKERS_PER_STREAM"


# ==========================================
# WORKER FUNCTION
# ==========================================
run_stream() {
    local stream_id=$1
    local start_offset=$2
    
    echo "🧵 [Stream $stream_id] Online. Offset: $start_offset"

    for (( i=$start_offset; i<$END_GLOBAL; i+=$TOTAL_PARALLEL_STREAMS )); do
        
        CURRENT_START=$i
        CURRENT_END=$((i + 1))
        
        RUN_ID="p${CURRENT_START}"
        LOG_FILE="logs/parallel/problem_${CURRENT_START}.log"
        mkdir -p logs/parallel

        echo "▶️  [Stream $stream_id] Starting Problem $CURRENT_START..."
        
        # ---------------------------------------------------------
        # ISOLATION: Managed by 'set -m'
        # ---------------------------------------------------------
        # We export the limit variable JUST for this command
        # We removed 'setsid' so 'wait' works correctly.
        COEVOLUTION_WORKERS=$WORKERS_PER_STREAM \
        uv run "$SCRIPT_PATH" \
            -r "$RUN_ID" \
            -s "$CURRENT_START" \
            -e "$CURRENT_END" \
            > "$LOG_FILE" 2>&1 &
        
        JOB_PID=$!
        
        # This now correctly blocks until the Python script finishes
        wait $JOB_PID
        
        EXIT_STATUS=$?
        
        if [ $EXIT_STATUS -eq 0 ]; then
            echo "✅ [Stream $stream_id] Problem $CURRENT_START Finished."
        else
            echo "❌ [Stream $stream_id] Problem $CURRENT_START Failed."
        fi
        
        # ---------------------------------------------------------
        # SURGICAL CLEANUP
        # ---------------------------------------------------------
        # Because of 'set -m', $JOB_PID is the Process Group Leader.
        # Adding the minus sign '-' kills the whole group.
        kill -9 -$JOB_PID 2>/dev/null
        
        sleep 2
    done
    
    echo "🏁 [Stream $stream_id] Complete."
}

# ==========================================
# MAIN ENTRY POINT
# ==========================================

echo "🚀 Launching $TOTAL_PARALLEL_STREAMS Parallel Streams..."

PIDS=()

for (( s=0; s<TOTAL_PARALLEL_STREAMS; s++ )); do
    STREAM_START_INDEX=$((START_GLOBAL + s))
    
    run_stream $s $STREAM_START_INDEX &
    
    pid=$!
    PIDS+=($pid)
    echo "   Started Stream $s (PID: $pid) starting at index $STREAM_START_INDEX"
done

echo "⏳ Manager waiting for streams..."

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "🎉 All Parallel Batches Complete!"