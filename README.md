# Single-view-contrastive-learning
for NBI laryngoscope image classification
## Running Environment
python 3.9.12<br>
pytorch 1.12.1<br>
cuda 11.3<br>
## Dataset
dataset<br>
--train<br>
----NT<br>
------001<br>
------002<br>
------...<br>
----SD<br>
----...<br>
--test<br>
----...<br>
## Training the Model
* train*.py: training file in different methods.<br>
* myDatasets.py: Data loading file.<br>
* model_selfcontrast_***: Use *** as backbone's single-view contrastive learning model.<br>
* loss: The folder of loss function.<br>
* gradcam: The folder of the lesion location module.<br>
* store: Training results save folder.<br>
## Model architecture
![Model architecture](https://raw.githubusercontent.com/hans-bbt/single-view-contrastive-learning/master/self_contrast_overall.jpg)
