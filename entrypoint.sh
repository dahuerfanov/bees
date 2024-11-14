#!/bin/bash
LR=${1:-0.0004}  # Default to 0.0004 if no argument is provided
BS=${2:-16} # Default to 16 if no argument is provided
EPOCHS=${3:-20} # Default to 20 if no argument is provided

echo "Received learning_rate: $LR"
echo "Received batch_size: $BS"
echo "Received epochs: $EPOCHS"

python3 tools/prepare_data_from_file_lists.py --data_root /data
if [ $? -eq 0 ]; then
    python3 run_bee_exp.py --dataset_root /data --learning_rate "$LR" --batch_size "$BS" --epochs "$EPOCHS" --torchscript_output /trained_models/model.pt
fi