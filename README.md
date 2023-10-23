# single-view-contrastive-learning
for NBI laryngoscope image classification
## dataset
dataset
-train
  -NT
    -001
    -002
    -...
  -SD
  -...
-test
  -...
## Training the Model
train*.py:training file in different methods.
myDatasets.py:Data loading file.
model_selfcontrast_***:Use *** as backbone's single-view contrastive learning model.
loss:The folder of loss function.
gradcam:The folder of the lesion location module.
store:Training results save folder.
## Model architecture
![Model architecture]([https://www.example.com/image.jpg](https://raw.githubusercontent.com/hans-bbt/single-view-contrastive-learning/master/self_contrast_overall.jpg)https://raw.githubusercontent.com/hans-bbt/single-view-contrastive-learning/master/self_contrast_overall.jpg)
