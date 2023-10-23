#Display the heat map

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize, \
    RandomAutocontrast, RandomEqualize  # , InterpolationMode
from torchvision.transforms import RandomResizedCrop, Resize, RandomAffine, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip,RandomAdjustSharpness,ColorJitter
import torch.utils.data as data
from os import listdir
from os.path import exists, join
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
import traceback as trace
import cv2
import numpy as np

image_size= 224

def myDataset(dest):
    if not exists(dest):
        print("dataset not exist ")
    return dest


def input_transform(data_type):  # need to add data augmentation
    # normal :  [0.6450,0.4089,0.3898],[0.0593,0.0375,0.0365]
    # NBI :     [0.4208,0.3874,0.3595],[0.0543,0.0493,0.0429]
    # all :     [0.4964,0.3987,0.3726],[0.0683,0.0464,0.0415]
    s=trace.extract_stack()
    if data_type == 'normal':
        return Compose([ToTensor(),
                        Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),

                        Normalize([0.6450, 0.4089, 0.3898], [0.0593, 0.0375, 0.0365])])
    elif data_type == 'NBI':
        return Compose([ToTensor(),
                        Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        RandomAutocontrast(),
                        Normalize([0.4208, 0.3874, 0.3595], [0.0543, 0.0493, 0.0429])])

def my_transform():  # need to add data augmentation

        return Compose([Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                       ])
def valid_input_transform(data_type):  # need to add data augmentation
    # normal :  [0.6450,0.4089,0.3898],[0.0593,0.0375,0.0365]
    # NBI :     [0.4208,0.3874,0.3595],[0.0543,0.0493,0.0429]
    # all :     [0.4964,0.3987,0.3726],[0.0683,0.0464,0.0415]
    s=trace.extract_stack()
    if data_type == 'normal':
        return Compose([Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                        Normalize([0.6450, 0.4089, 0.3898], [0.0593, 0.0375, 0.0365])])
    elif data_type == 'NBI':
        return Compose([ToTensor(),
                        Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),
                        #RandomAffine(degrees=30,translate=(0.3, 0.2)),
                        #RandomRotation(degrees=90),
                        #RandomHorizontalFlip(p=0.5),
                        #RandomVerticalFlip(p=0.5),
                        #RandomAutocontrast(),
                        #ColorJitter(brightness=0.5),

                        Normalize([0.4208, 0.3874, 0.3595], [0.0543, 0.0493, 0.0429])])


#    return Compose([Resize((image_size,image_size), interpolation=2),
#                    RandomAffine(50),RandomRotation(45),RandomHorizontalFlip(p=1),RandomVerticalFlip(p=1),
#                    ToTensor(),Normalize([0.4121,0.3773,0.3577],[0.0543,0.0503,0.0447])]) 
#    return Compose([RandomResizedCrop(image_size,scale=(1.0,1.0),ratio=(1.0,1.0),interpolation=2),
#                    RandomAffine(50),RandomRotation(45),RandomHorizontalFlip(p=0.5),RandomVerticalFlip(p=0.5),
#                    ToTensor(),Normalize([0.4121,0.3773,0.3577],[0.0543,0.0503,0.0447])])

def target_transform():
    return Compose([ToTensor()])

def get_training_set(dest, classes, data_type, color_dim=3):
    root_dir = myDataset(dest)
    train_dir = join(root_dir, "train")
    return TrainDatasetFromFolder(train_dir, classes, data_type, color_dim,
                                  input_transform=input_transform(data_type))

def get_test_set(dest, classes, data_type,color_dim=3):
    root_dir = myDataset(dest)
    test_dir = join(root_dir, "valid")
    return TestDatasetFromFolder(test_dir, classes, data_type, color_dim,
                                 input_transform=valid_input_transform(data_type,))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, data_transform,class_name):
    img = Image.open(filepath).convert('RGB')
    img=img.resize((224,224))
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)

    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    classes = ['health', 'lca', 'mild', 'prelca', 'moderate', 'severe']
    target_category = classes.index(class_name)  # label

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    print((filepath))

    image_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imshow("heatmap", image_bgr)
    cv2.waitKey(100)
    img_name=filepath.strip().split("\\")
    cv2.imwrite(join("heatmap_img",img_name[-5]+'_'+img_name[-4]+'_'+img_name[-3]+'_'+img_name[-2]+'_'+img_name[-1]),image_bgr)


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, class_dir, classes, data_type, color_dim=3, input_transform=None):
        super(TrainDatasetFromFolder, self).__init__()
        self.class_dir = class_dir
        self.classes = classes
        self.data_type = data_type
        self.color_dim = color_dim
        self.input_transform = input_transform

        class_names = [x for x in listdir(class_dir)]

        path_normal = []
        path_NBI = []
        self.image_files_normal = []
        self.image_files_NBI = []
        self.image_files = []
        for name_class in class_names:
            patients = [x for x in listdir(join(class_dir, name_class))]
            for pat in patients:
                path_NBI.append(join(class_dir, name_class, pat, 'NBI'))
        for index in path_normal:
            image_names = [x for x in listdir(index)]
            for image in image_names:
                self.image_files_normal.append(join(index, image))
        for index in path_NBI:
            image_names = [x for x in listdir(index)]
            for image in image_names:
                self.image_files_NBI.append(join(index, image))

    def __getitem__(self, index):
        self.image_files = self.image_files_NBI
        class_name = self.image_files[index].strip().split("\\")[-4]  # 类型名称
        load_img(self.image_files[index], self.input_transform,class_name)
        return index

    def __len__(self):
        if self.data_type == 'normal':
            return len(self.image_files_normal)
        elif self.data_type == 'NBI':
            return len(self.image_files_NBI)
        # return len(self.image_files)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, class_dir, classes, data_type, color_dim, input_transform=None):
        super(TestDatasetFromFolder, self).__init__()
        self.class_dir = class_dir
        self.classes = classes
        self.data_type = data_type
        self.color_dim = color_dim
        self.input_transform = input_transform
        self.my_transform=my_transform()

        class_names =  [x for x in listdir(class_dir)]
        path = []
        self.image_files = []
        for name_class in class_names:
            patients = [x for x in listdir(join(class_dir, name_class))]
            for pat in patients:
                path.append(join(class_dir, name_class, pat, 'NBI'))
        for index in path:
            image_names = [x for x in listdir(index)]
            for image in image_names:
                self.image_files.append(join(index, image))

    def __getitem__(self, index):
        self.image_files = self.image_files
        class_name = self.image_files[index].strip().split("\\")[-4]  # 类型名称
        load_img(self.image_files[index], self.input_transform,class_name)

        return index

    def __len__(self):
        return len(self.image_files)


from utils_heatmap import GradCAM, show_cam_on_image
from torch.utils.data import DataLoader

classes=['health','lca','mild','prelca','moderate','severe']

traindataset =get_training_set(myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
testdataset =get_test_set(myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
train_loader=DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
test_loader=DataLoader(testdataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

import model_repAlexnet
model = model_repAlexnet.RepAlexNet(num_blocks=[1, 1, 1, 1],num_classes=6)
model = model_repAlexnet.repvgg_model_convert(model)
model.load_state_dict(torch.load("store\\repalexnet_heatmap_n100.pth"))

print("model load complete!")

target_layers = [model.stage5]

grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

for i, (datas) in enumerate(test_loader,):
    pass
# for i, (datas) in enumerate(train_loader,):
#     pass
