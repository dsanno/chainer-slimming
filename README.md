# Implementation of Network Slimming

Implementation of ["Learning Efficient Convolutional Networks through Network Slimming"](https://arxiv.org/abs/1708.06519).

# Requirements

* Python 2.7
* [Chainer 2.0.0](http://chainer.org/)
* [Cupy 1.0.0](http://docs.cupy.chainer.org/en/stable/)
* [matplotlib](http://matplotlib.org/)

# Example

## Train model

```
$ python src/train.py model/base.json -p vgg_1 -g 0 --no-valid-data
```

## Prune

```
$ python src/prune.py model/vgg_1.model model/base.json model/vgg_2_org.model model/vgg_2.json 0.5
```

## Fine tune

```
$ python src/train.py model/vgg_2.json -g 0 -m model/vgg_2_org.model --lambda-value 0
```

# How to use

## Train

```
$ python src/train.py <structure_path> -g <gpu_id> -m <model_path> -b <batch_size> -p <prefix> --optimizer <optimizer> --epoch <epoch_num> --lr-decay-epoch <lr_decay>
```

While training the following files are saved.
* model parameter file `<prefix>.model`
* loss and error log file `<prefix>_log.csv`
* loss curve image `<prefix>_loss.png`
* error curve image `<prefix>_error.png`

Parameters:

* `<structure_path>`: Required  
Model structure JSON file path (e.g. model/base.json)
* `-g (--gpu) <int>`: Optional  
GPU device ID. Negative value indicates CPU (default: -1)
* `-m (--model) <model name>`: Optional  
Model file path (default: None)
* `-b (--batch_size) <int>`: Optional  
Mini batch size (default: 64)
* `-p (--prefix) <str>`: Optional  
Prefix of saved model file. (default: file name of structure JSON except extension)
* `--base <int>`: Optional  
Base size of model (default: 64)
* `--epoch <int>`: Optional  
Training epoch (default: 160)
* `--save-epoch <int>`: Optional  
Epoch interval to save model parameter file. 0 indicates model paramete is not saved at fixed intervals. Note that the best accuracy model is always saved even if this parameter is 0. (default: 0)
* `--optimizer <str>`: Optional  
Optimizer name (`sgd` or `adam`, default: sgd)
* `--lr <float>`: Optional  
Initial learning rate for SGD (default: 0.1)
* `--alpha <float>`: Optional  
Initial alpha for Adam (default: 0.001)
* `--lr-decay-epoch <int>`: Optional  
Epoch interval to decay learning rate. Learning rate is decay to 1/10 at this intervals. (default: 80,120)
* `--weight-decay <float>`: Optional  
Weight decay (default: 0.0001)
* `--lambda-value <float>`: Optional  
Regularization factor for gamma of batch normalization. It is recommended to set 0 for fine tuning. (default: 0.0001)
* `--no-valid-data`: Optional  
If set validation data is not used (i.e. train data is not separated to train/validation).
* `--seed <int>`: Optional  
Random seed (default: 1)

Example:
```
$ python src/train.py model/base.json -g 0 -b 64 -p vgg_1 --optimizer sgd --epoch 160 --lr-decay-epoch 80,120
```

## Prune

```
$ python src/prune.py <src_model_path> <src_structure_path> <dest_model_path> <dest_structure_path> <prune_ratio>
```

Parameters:

* `<src_model_path(str)`: Required  
Source model file path
* `<src_structure_path>(str)`: Required  
Source structure JSON file path
* `<dest_model_path(str)`: Required  
Destination model file path
* `<dest_structure_path>(str)`: Required  
Destination structure JSON file path
* `<prune ratio>(float)`: Required
Prune ratio

Example:
```
$ python src/prune.py model/vgg_1.model model/base.json model/vgg_2_org.model model/vgg_2.json 0.5
```

# License

MIT License
