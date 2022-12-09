
# Scribble2D5
Scribble2D5: Weakly-Supervised Volumetric Image Segmentation via Scribble Annotations

# environment

```
python 3.6.9
torch 1.8.0+cu111
torchvision 0.9.0+cu111
monai 0.4.0
```

# MONAI
Since our Predictor has multiple outputs, the sliding window inference function of the MONAI should be added with the index hyperparameter.

