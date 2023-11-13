# Single-view-contrastive-learning
for NBI laryngoscope image classification
## Running Environment
```
python 3.9.12
pytorch 1.12.1
cuda 11.3
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

### Data loading
```
myDatasets.py: Data loading file
```

### Training Model
```
train***.py #training with backbone ***
```
