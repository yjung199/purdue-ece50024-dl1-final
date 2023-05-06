# Optimization as a Model for Few-shot Learning
Pytorch replication study of [Optimization as a Model for Few-shot Learning](https://openreview.net/forum?id=rJY0-Kcll) in ICLR 2017 (Oral)

## Prerequisites
- Python 3.0+ (used 3.8)
- Pytorch (developed on 2.0 with cuda version 12.0)
- Conda environment (yml file downloadable from [here](https://purdue0-my.sharepoint.com/:u:/g/personal/jung199_purdue_edu/EXcelSEWGo5Gt4291iTlEEIBxClTEeZarKbU1vMNkBic6w?e=4ozDQC))
  - Use `conda env create -f environment.yml` to create the environment

## Dataset
- MiniImagenet as described in the original paper. 
  - Downloadable from [here](https://purdue0-my.sharepoint.com/:u:/g/personal/jung199_purdue_edu/EdI7dATMbHVNp7QILyhn5YsB3i3BHq6glKEOYOZblTTzpw?e=Adbwus) (~2.51GB, One drive link)
  - Put it directly under the root directory of this repo.


## Prerequisites
- Download the MiniImagenet and put it **directly under the root directory** of this proejct.
- Check out `scripts/train_5s_5c`, make sure `--data_root` is properly set

## Train/Test the Model
The script to train and test the model is provided. (Beware that for Linux, I made .sh file but it is untested.)
From the root directory of this project, run
```bash
# For Windows
scripts/train-5-shot-5-class
# For Linux
bash scripts/train-5-shot-5-class
```
to run 5-shot, 5-class training. 

To evaluate, run 
```bash
# For Windows
scripts/test-5-shot-5-class
# For Linux
bash scripts/test-5-shot-5-class
```
Note: when evaluating, remember to change `--resume` and `--seed` arguments in the file.

## Development Environment
- Windows 10 Pro 64-bit (10.0, Build 19042)
- Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (16 CPUs), ~2.6GHz
- 24GB RAM
- NVIDIA Tesla V100-PCIE-16GB


## References
- [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) (Data loader)
- [meta-learning-lstm](https://github.com/twitter/meta-learning-lstm) (Author's repo in Lua)

