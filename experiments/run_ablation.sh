#!/bin/bash
set -e

echo "================================"
echo "BloodMNIST H2 Ablation"
echo "================================"

# 50% data: pretrained + scratch, seeds 0-2
for seed in 0 1 2
do
  python src/train.py \
    --dataset blood \
    --condition pretrained \
    --seed $seed \
    --fraction 0.50
done

for seed in 0 1 2
do
  python src/train.py \
    --dataset blood \
    --condition scratch \
    --seed $seed \
    --fraction 0.50
done


# 25% data: pretrained + scratch, seeds 0-2
for seed in 0 1 2
do
  python src/train.py \
    --dataset blood \
    --condition pretrained \
    --seed $seed \
    --fraction 0.25
done

for seed in 0 1 2
do
  python src/train.py \
    --dataset blood \
    --condition scratch \
    --seed $seed \
    --fraction 0.25
done


# 10% pretrained:
# seed 0 already completed successfully.
for seed in 1 2
do
  python src/train.py \
    --dataset blood \
    --condition pretrained \
    --seed $seed \
    --fraction 0.10
done


# 10% scratch, seeds 0-2
for seed in 0 1 2
do
  python src/train.py \
    --dataset blood \
    --condition scratch \
    --seed $seed \
    --fraction 0.10
done

echo "================================"
echo "ABLATION COMPLETE"
echo "================================"
