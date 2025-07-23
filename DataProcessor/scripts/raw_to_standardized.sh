DATASET=$1
NUM_SAMPLES=${2:-"-1"}

python -m gui_agent_data.stage.raw_to_standardized $DATASET datasets/$DATASET/raw_trajs datasets/$DATASET/standardized --num_samples $NUM_SAMPLES
