#!/bin/bash

MODEL_REPO=${1:-"meta-llama/Llama-3.3-70B-Instruct"} # Default if $1 is missing
MODEL_ALIAS=${2:-"llama70b"}                          # Default if $2 is missing
ENV_NAME=${3:-"minihack"}                             # Default if $3 is missing
SWEEP_DIR="multirun/${MODEL_ALIAS}_${ENV_NAME}"

# 2. Validation (Optional: print what is running)
echo "=================================================="
echo "CONFIGURING RUN"
echo "  Model Repo : $MODEL_REPO"
echo "  Alias      : $MODEL_ALIAS"
echo "  Env Name   : $ENV_NAME"
echo "  Output Dir : $SWEEP_DIR"
echo "=================================================="



PORT=8080

COMMON_OVERRIDES="client.client_name=vllm \
client.model_id=$MODEL_REPO \
client.base_url=http://localhost:$PORT/v1 \
eval.num_workers=16 \
envs.names=$ENV_NAME \
eval.num_episodes.pogs=100 \
eval.num_episodes.minihack=100 \
eval.num_episodes.crafter=100 \
eval.num_episodes.textworld=33"

echo "--------------------------------------------------"
echo "Starting Experiment 1: Sweep plan_k (2,3,4,6,8,12,16,24)"
echo "--------------------------------------------------"

# Hydra Multirun (-m) for the K values
python eval.py -m \
  $COMMON_OVERRIDES \
  hydra.sweep.dir=$SWEEP_DIR/sweep_plan_k \
  agent.type=plan_every_k_step \
  agent.plan_k=2,3,4,6,8,12,16,24

echo "--------------------------------------------------"
echo "Starting Experiment 2: Baselines (always_plan, never_plan)"
echo "--------------------------------------------------"

# Hydra Multirun (-m) for the agent types
python eval.py -m \
  $COMMON_OVERRIDES \
  hydra.sweep.dir=$SWEEP_DIR/baselines \
  agent.type=always_plan,never_plan

echo "JOB DONE: $(date)"