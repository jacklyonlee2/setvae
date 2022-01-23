# SetVAE
This repository implements a drastically simplified version of [SetVAE: Learning Hierarchical Composition for Generative Modeling of Set-Structured Data](https://arxiv.org/abs/2103.15619)
while achieving the same performance as the official implementation.

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate setvae
```

* Compile CUDA extensions.

```shell
sh scripts/install.sh
```

* Download ShapeNet dataset and trained checkpoints.

```shell
sh scripts/download.sh
```

## Training
* You can train SetVAE using `train.py` or provided scripts.

```shell
# Train SetVAE using CLI
python train.py --name hello --cate airplane
# Train SetVAE using provided settings
sh scripts/train_shapenet_aiplane.sh
sh scripts/train_shapenet_car.sh
sh scripts/train_shapenet_chair.sh
```

## Testing
* You can evaluate checkpointed models using `test.py` or provided scripts.
```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path hello.pth --cate car
# Test provided SetVAE checkpoints
sh scripts/test_shapenet_aiplane.sh
sh scripts/test_shapenet_car.sh
sh scripts/test_shapenet_chair.sh
```
