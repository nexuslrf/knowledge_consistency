# Knowledge Isomorphism

Here are codes supporting the experiments in the submitted paper.

### Environment Setup:

* python 3.6
* pytorch 1.0
* tensorboard
* jupyter notebook

### Get Dataset:



### Training the classification net:

All the big classification nets can be trained via the following scripts:

E.g. to train a vgg16_bn on CUB200 dataset:

```bash
python Training.py --device_ids [0,1] --lr 0.01 --epochs 300 --dataset CUB200 --save_epoch 50 --suffix lr-2_sd0 --seed 0 --batch-size 128 --epoch_step 60 --arch vgg16_bn
```

### Training Simple Trans-Net:

This Trans-Net is to learn the isomorphism information among different neural networks. i.e. We use a multi-layer neural network to reconstruct the target feature maps (Net B) from the source feature maps (Net A).

To save the RAM consumption, we first generate all feature maps of specified conv_layer of all training images in the dataset. Like the following scripts:

Note: You need train 2 classification networks with the same arch to run `ConvOutput.py`

```bash
python ConvOutput.py --arch vgg16_bn --batch_size 128 --resume1 [Net A] --resume2 [Net B] --dataset CUB200 --conv_layer 30
```

All the trans-Nets can be trained via the following scripts:

```bash
python Training_TransNet.py --arch vgg16_bn --device_ids [0,0] --dataset CUB200 --conv_layer 30 --convOut_path [feature map path] --lr 0.0001 --alpha [0.1,0.1] --epochs 1000 --suffix a0.1_lr-4
```

### Trans-Classification:

To take full use of trans-net, we further finetune the rest layers after trans-net for target classification network.  

```bash
python transClassifier.py --arch vgg16_bn --net_A [Net A] --net_B [Net B] --resume_Ys [Trans-Net] --dataset CUB200 --gpu 0 --conv_layer 30 --epochs 1000 --lr 0.00001 --logspace 2 --suffix lr-5_lg2
```

### Visualization:

We write a simple [jupyter notebook](link!) to visualize the original image, corresponding feature maps , learnt feature maps, different fuzzy level sub-feature map, etc.

