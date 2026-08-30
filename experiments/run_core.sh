#!/bin/bash
set -e

for seed in 1 2 3 4
do
  python src/train.py \
    --dataset pneumonia \
    --condition pretrained \
    --seed $seed
done

for seed in 0 1 2 3 4
do
  python src/train.py \
    --dataset pneumonia \
    --condition scratch \
    --seed $seed
done