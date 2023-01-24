#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --time=80:00:00
export OMP_NUM_THREADS=1

source activate ic3net


python -u pp_loc/main.py \
  --env_name pp \
  --nagents 7 \
  --max_steps 40 \
  --nprocesses 1 \
  --num_epochs 300 \
  --epoch_size 10 \
  --hid_size 128 \
  --entr 0.005 \
  --gamma 0.99 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.00025 \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --learn_second_graph \
  --first_gat_normalize \
  --second_gat_normalize \
  --save \
  --seed 2022