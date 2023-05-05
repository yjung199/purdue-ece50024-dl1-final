param(
    [string]$mode = "train",
    [int]$nShot = 5,
    [int]$nEval = 15,
    [int]$nClass = 5,
    [int]$inputSize = 4,
    [int]$hiddenSize = 20,
    [float]$lr = 0.001,
    [int]$episode = 50000,
    [int]$episodeVal = 100,
    [int]$epoch = 8,
    [int]$batchSize = 25,
    [int]$imageSize = 84,
    [float]$gradClip = 0.25,
    [float]$bnMomentum = 0.95,
    [float]$bnEps = 0.001,
    $dataRoot = "data/miniImagenet/",
    $pinMem = "True",
    $logFreq = 50,
    $valFreq = 1000
)

python main.py --mode $mode `
               --dataset miniimagenet `
               --data_root data/miniImagenet/ `
               --num_shot $nShot `
               --num_eval $nEval `
               --num_class $nClass `
               --input_size $inputSize `
               --hidden_size $hiddenSize `
               --lr $lr `
               --episode $episode `
               --episode_val $episodeVal `
               --epoch $epoch `
               --batch_size $batchSize `
               --image_size $imageSize `
               --grad_clip $gradClip `
               --bn_momentum $bnMomentum `
               --bn_eps $bnEps `
               --pin_mem $pinMem `
               --log_freq $logFreq `
               --val_freq $valFreq