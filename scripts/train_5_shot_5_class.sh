#!/bin/bash

mode="train"
nShot=5
nEval=15
nClass=5
inputSize=4
hiddenSize=20
lr=0.001
episode=10000
episodeVal=100
epoch=8
batchSize=25
imageSize=84
gradClip=0.25
bnMomentum=0.95
bnEps=0.001
dataRoot="data/miniImagenet/"
pinMem="True"
logFreq=50
valFreq=1000

python main.py --mode "$mode" \
               --dataset miniimagenet \
               --data_root "$dataRoot" \
               --num_shot "$nShot" \
               --num_eval "$nEval" \
               --num_class "$nClass" \
               --input_size "$inputSize" \
               --hidden_size "$hiddenSize" \
               --lr "$lr" \
               --episode "$episode" \
               --episode_val "$episodeVal" \
               --epoch "$epoch" \
               --batch_size "$batchSize" \
               --image_size "$imageSize" \
               --grad_clip "$gradClip" \
               --bn_momentum "$bnMomentum" \
               --bn_eps "$bnEps" \
               --pin_mem "$pinMem" \
               --log_freq "$logFreq" \
               --val_freq "$valFreq"