# Instructions
1. Install all dependencies in `requirements.txt`
2. Run the following command in terminal:
```commandline
python train.py \                                                                                
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu
```