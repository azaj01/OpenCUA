DATASET=$1
NUM_SAMPLES=${2:-1}
python -m gui_agent_data.stage.extract_raw $DATASET datasets/$DATASET/raw_trajs -n $NUM_SAMPLES