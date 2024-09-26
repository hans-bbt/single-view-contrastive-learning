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
我们上传了三种不同骨干网络的自对比模型的代码实现，你可以替换这些代码中的骨干网络以实现你自己的自对比模型。
```
model_selfcontrast_alexnet.py
model_selfcontrast_swin.py
model_selfcontrast_vgg11.py
```

### Training Model
三种不同的训练方式，在这三种方法中，我们使用了不同的对比的对象，train0.py中的对比对象是热力区域最大值和随机生成区域，train1.py中对比的对象是热力值最大区域和热力值第二大区域，train2.py中对比的对象是连个随机生成的区域。
```
train0.py #training with different methods
train1.py #training with different methods
train2.py #training with different methods
```

### Validate trained model
验证训练的结果，因为我们仅在训练过程中使用了对比学习因此上面的三种训练方法可以使用同样的代码验证，这段代码会输出预测错误的情况和总体准确率。
```
valid.py #training with different methods
```

### Loss function
不同的损失值计算方式会对结果产生影响，因此我们实现了两种不同的损失函数，并通过后续的实验对比那种更好。损失函数由两部分组成，交叉熵损失和对比损失，这两个损失函数不同的地方在于他么对比损失的实现。loss/loss_selfcontrast0.py实现了patch之间的对比损失计算，loss/loss_selfcontrast1.py实现了整张图像和两个patch之间的对比损失之计算。
```
loss/loss_selfcontrast0.py
loss/loss_selfcontrast1.py
```

### HeatMap generation
```
gradcam/model_repAlexnet.py #repAlexNet网络
gradcam/show_heatmap.py #展示热力图
gradcam/train_modelheatmap.py #训练repAlexNet网络
gradcam/utils_heat_map.py #生成热力图的辅助函数
gradcam/utils_max1.py #计算生成热力图的热力值最大区域
gradcam/utils_max2.py #计算生成热力图的热力值最大的两个区域
gradcam/write_max1.py #将utils_max1.py的结果保存到硬盘
gradcam/write_max2.py #将utils_max2.py的结果保存到硬盘
```
## Dataset
所有数据如下图所示：

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
