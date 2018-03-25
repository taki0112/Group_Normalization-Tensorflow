# Group_Normalization-Tensorflow
Simple Tensorflow implementation of "Group Normalization"

## Usage
```python
from ops import *
  x = conv(x)
  x = group_norm(x) 
```

## Normalization function
![norm](./assests/norm.png)

## ImageNet Results
### classification error per batch sizes
![bn_gn](./assests/bn_gn.png)

### Comparison of error curves with a batch size of 32 (ResNet 50)
![error](./assests/error.png)

### Sensitivity to batch sizes (ResNet 50)
![batch_size](./assests/batch_size.png)

## COCO Results
![coco](./assests/coco.png)

## Author
Junho Kim
