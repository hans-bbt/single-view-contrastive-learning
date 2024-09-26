# Single-view-contrastive-learning
for NBI laryngoscope image classification
## Running Environment
```
python 3.9.12
pytorch 1.12.1
cuda 11.3
argparse
csv
tqdm
timm
cv2
traceback
PIL
numpy
```


### Data loading
```
myDatasets.py: Data loading file
```
### Model implementation
```
model_selfcontrast_alexnet.py
model_selfcontrast_swin.py
model_selfcontrast_vgg11.py
```

### Training Model
```
train0.py #training with different methods
train1.py #training with different methods
train2.py #training with different methods
```

### Validate trained model
```
valid.py #training with different methods
```

### Loss function
```
loss/loss_selfcontrast0.py
loss/loss_selfcontrast1.py
```

### HeatMap generation
```
loss/loss_selfcontrast0.py
loss/loss_selfcontrast1.py
```
## Dataset
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
