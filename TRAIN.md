# Install docker
# Install nvidia-docker
- Hướng dẫn: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

# Pull image
- Tensorflow
```docker pull tensorflow/tensorflow:latest-gpu``` (https://hub.docker.com/r/tensorflow/tensorflow/tags)
- Pytorch
```docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime``` (https://hub.docker.com/r/pytorch/pytorch/tags)
# Run train
- Tensorflow
```docker run --gpus all -v $PWD:/tmp -w /tmp -it tensorflow/tensorflow:latest-gpu <<<python train.py>>>```
- Pytorch
```docker run --gpus all -v $PWD:/tmp -w /tmp -it pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime <<<python train.py>>>```