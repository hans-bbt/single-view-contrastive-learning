# Single-view-contrastive-learning
for NBI laryngoscope image classification
## Requirments
```
python 3.9.12
pytorch 1.12.1
cuda 11.3
numpy
PIL
cv2
timm
argparse
csv
tqdm
traceback
```

### Data loading
```
myDatasets.py: Data loading file
```
### Model implementation
We have uploaded three self-contrasting models, which differ in that they use different backbone networks, and you can replace the backbone networks in these codes to implement your own self-contrasting model.
```
model_selfcontrast_alexnet.py
model_selfcontrast_swin.py
model_selfcontrast_vgg11.py
```

### Training Model
Three different training methods. In these three methods, we use different comparison objects. In train0.py, the comparison objects are the maximum heatmap value area and the randomly generated area, and in train1.py, the comparison objects are the maximum heatmap value area and the second largest heatmap value area. train2.py takes two randomly generated regions as input to the loss function.
```
train0.py #training with different methods
train1.py #training with different methods
train2.py #training with different methods
```

### Validate trained model
Verify the results of the training, because we only used contrastive learning during the training process so the above three training methods can be verified using the same way, which outputs the cases of incorrect prediction and the overall accuracy.
```
valid.py #training with different methods
```

### Loss function
Different ways of calculating the loss value can affect the results, so we implement two different loss functions and compare the performance between the two through experiments. The loss function consists of two parts, the cross-entropy loss and the contrast loss. The difference between the two loss functions is the contrast loss. loss/loss_selfcontrast0.py realizes the calculation of contrast loss between patches, and loss/loss_selfcontrast1.py realizes the calculation of contrast loss between the whole image and two patches.
```
loss/loss_selfcontrast0.py
loss/loss_selfcontrast1.py
```

### HeatMap generation
```
gradcam/model_repAlexnet.py # repAlexNet network
gradcam/show_heatmap.py # shows the heatmap
gradcam/train_modelheatmap.py # Train the repAlexNet network
gradcam/utils_heat_map.py # Function for generating heat maps
gradcam/utils_max1.py # calculates the maximum heatmap value value region of the generated heat map
gradcam/utils_max2.py # computes the two regions with the largest heatmap value value for generating the heat map
```
## Dataset
All data are shown below, and the six categories are (a) Normal Tissues, (b) Inflammatory Keratosis, (c) Mild Dysplasia, (d)Moderate Dysplasia, (e) Severe Dysplasia, (f) Squamous Cell Carcinoma.：
![Image text](https://github.com/hans-bbt/single-view-contrastive-learning/blob/master/NBI_six_classes.png)
dataset directory tree:
```
├─NBI laryngoscopy images dataset
│  ├─train
│  │  ├─normal tissues
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─inflammatory keratosis
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─mild dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─moderate dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─severe dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─squamous cell carcinoma
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  ├─valid
│  │  ├─normal tissues
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─inflammatory keratosis
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─mild dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─moderate dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─severe dysplasia
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
│  │  ├─squamous cell carcinoma
│  │  │  │  001
│  │  │  │  │  xxx.jpg
│  │  │  │  │  ...
│  │  │  │  002
│  │  │  │  ...
```
