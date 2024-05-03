# To Uncertainty and Beyond: Enhancing Self-Supervised Learning in Computer Vision

## Introduction 
This repository contains the code base for the master's thesis "To Uncertainty and Beyond: Enhancing Self-Supervised Learning in Computer Vision".

### Abstract
Not accuracy matters, reliability does—the performance of a model should not solely
be measured by its average accuracy, but also by its ability to provide reliable predic-
tions. Lack of robustness is a major challenge, especially in critical domains such as
medicine and aviation. While well-established methods such as Monte Carlo (MC)
Dropout or Deep Ensembles exist to quantify uncertainty in supervised tasks, the
literature on uncertainty estimation in self-supervised learning (SSL) remains lim-
ited.
This work aims to fill this gap by introducing a probabilistic encoder for uncer-
tainty estimation in self-supervised contrastive learning for computer vision. Our
goal is to determine whether a learned representation effectively captures the seman-
tic properties of an image. By using a probabilistic encoder, we can use the dispersion
of the modelled distribution as a measure of uncertainty. The potential applications
of this approach are diverse, including sample selection for medical image annotation
and active learning pipelines.

## Usage
To train models, run the `main.py` file with the appropriate command-line arguments. 
Here's an example command to run probabilistic SimCLR with PowerSpherical distribution, a ResNet18 bacbone and a BatchSize of 512 and 1000 Epochs training lentgh:
```bash
python3 /path/to/UASSL/main.py -m "SimCLR" --loss "NT-Xent" --distribution "powerspherical" --network "resnet18" --batch_size 512 --epochs 1000
```

## Acknowledgements

This project builds upon the work of Florian Schulz from TU Berlin [Link to the GitHub repository](https://github.com/FloCF/TorchSelfSup)
