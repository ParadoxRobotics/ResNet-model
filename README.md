# ResNet-model
Simple ResNet model trained on CIFAR10 in pytorch:
- CIFAR10_ResNet.py, is custom made ResNet34-like network.
- ResNet_W.py, is a ResNet34 network with SELU activation (Conv/linear), Alphadrop dropout (for linear layer) and custom weight init. The aim is to train a ResNet34 network with no Batch Normalization and batch_size = 1 with SGD (incremental training fashion).
