#!/bin/bash
python benchmark.py \
--batch-size 4 \
--seq-len 1000 \
--d-model 64 \
--d-feat 32 \
--nhead 8 \
--num-layers 4 \
--dim-feedforward 512 \
--dropout 0.0 \
--n-emitters 5 \
--margin 1.9 \
--max-triplet-n 512 \
--min-cluster-size 20 \
--warmup 3 \
--repeats 10 \
--train-windows 329815 \
--val-windows 34715 \
--epochs 5 \
--device xla
