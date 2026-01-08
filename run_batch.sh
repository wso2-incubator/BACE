#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_PATH="experiments/scripts/coevolution/run_coevolution.py"
START_GLOBAL=0
END_GLOBAL=40
STEP_SIZE=5

# ==========================================
# HELPER FUNCTIONS
# ==========================================

cleanup_lingering_processes() {
    echo "🧹 [CLEANUP] Scanning for lingering workers..."
    
    # We grep for the script name to ensure we don't kill system python processes
    # We exclude 'grep' itself and the current bash script
    pids=$(pgrep -f "run_coevolution.py")
    
    if [ -n "$pids" ]; then
        echo "⚠️  Found zombie processes: $pids"
        echo "🔪 Killing..."
        # Soft kill first
        kill $pids 2>/dev/null
        sleep 2
        # Hard kill if still alive
        kill -9 $pids 2>/dev/null
        echo "✅ Cleanup complete."
    else
        echo "✨ No lingering processes found."
    fi
}

# ==========================================
# MAIN LOOP
# ==========================================

# Iterate from 0 to 35 (since 40 is the exclusive upper bound)
for (( i=$START_GLOBAL; i<$END_GLOBAL; i+=$STEP_SIZE )); do
    
    CURRENT_START=$i
    CURRENT_END=$((i + STEP_SIZE))
    
    RUN_ID="h${CURRENT_START}-${CURRENT_END}"
    LOG_FILE="${RUN_ID}.log"
    
    echo "=================================================="
    echo "🚀 STARTING BATCH: $RUN_ID (Index $CURRENT_START to $CURRENT_END)"
    echo "=================================================="
    
    # 1. Run the command in the background
    # We use 'uv run' as requested.
    nohup uv run "$SCRIPT_PATH" \
        -r "$RUN_ID" \
        -s "$CURRENT_START" \
        -e "$CURRENT_END" \
        > "$LOG_FILE" 2>&1 &
    
    # 2. Capture the Process ID (PID) of the most recent background job
    JOB_PID=$!
    echo "📝 Log file: $LOG_FILE"
    echo "⏳ Job PID: $JOB_PID - Waiting for completion..."
    
    # 3. Wait for this specific job to finish
    # This blocks the loop until the python script exits.
    wait $JOB_PID
    
    EXIT_STATUS=$?
    
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "✅ Batch $RUN_ID finished successfully."
    else
        echo "❌ Batch $RUN_ID failed with exit code $EXIT_STATUS."
        # Optional: exit 1 # Uncomment to stop entire script on failure
    fi
    
    # 4. SAFETY CLEANUP
    # Even if the main process finished, child workers might be stuck.
    # We kill them now to free up RAM for the next batch.
    cleanup_lingering_processes
    
    echo "😴 Resting for 5 seconds before next batch..."
    sleep 5

done

echo "🎉 All batches (0-$END_GLOBAL) complete!"