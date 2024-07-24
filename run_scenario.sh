#!/bin/bash

# Function to kill previous instances of run_step.sh
kill_previous_instance() {
    # Get the process ID of the previous instance
    PID=$(pgrep -f run_test.sh | grep -v $$)
    
    if [ ! -z "$PID" ]; then
        echo "Killing previous instance of run_step.sh with PID: $PID"
        kill -9 $PID
    else
        echo "No previous instance of run_step.sh found."
    fi
}

# Kill previous instances if any
kill_previous_instance

# Remove the contents of the results directory
RESULTS_DIR="/workspace/team_code/results"
if [ -d "$RESULTS_DIR" ]; then
    echo "Removing contents of the results directory: $RESULTS_DIR"
    rm -rf "${RESULTS_DIR:?}"/*
fi

#qdtrack_ training.xml
# export ROUTES=${LEADERBOARD_ROOT}/data/routes_controlling.xml
export ROUTES=/workspace/team_code/route_1_avddiem.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/carla_behavior_agent/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/carla_behavior_agent/config_agent_basic.json
export CHECKPOINT_ENDPOINT=${LEADERBOARD_ROOT}/results.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=193.205.163.183
export CARLA_PORT=6024
export CARLA_TRAFFIC_MANAGER_PORT=8824
export CHECKPOINT_ENDPOINT=/workspace/team_code/results/simulation_results.json
export DEBUG_CHECKPOINT_ENDPOINT=/workspace/team_code/results/live_results.txt
export RESUME=0
export TIMEOUT=60
# 193.205.163.183
# 193.205.163.17

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--timeout=${TIMEOUT} \
--traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT}
