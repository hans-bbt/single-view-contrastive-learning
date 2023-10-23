from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize, \
    RandomAutocontrast, RandomEqualize  # , InterpolationMode
from torchvision.transforms import RandomResizedCrop, Resize, RandomAffine, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip,RandomAdjustSharpness,ColorJitter
import torch.utils.data as data
from os import listdir
from os.path import exists, join
from PIL import Image
from torchvision.transforms import InterpolationMode
import traceback as trace


image_size= 1080

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
        return Compose([Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                        Normalize([0.6450, 0.4089, 0.3898], [0.0593, 0.0375, 0.0365])])
    elif data_type == 'NBI':
        return Compose([Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        RandomAutocontrast(),
                        ToTensor(),
                        Normalize([0.4208, 0.3874, 0.3595], [0.0543, 0.0493, 0.0429])])
def test_input_transform(data_type):  # need to add data augmentation
    # normal :  [0.6450,0.4089,0.3898],[0.0593,0.0375,0.0365]
    # NBI :     [0.4208,0.3874,0.3595],[0.0543,0.0493,0.0429]
    # all :     [0.4964,0.3987,0.3726],[0.0683,0.0464,0.0415]
    s=trace.extract_stack()
    if data_type == 'normal':
        return Compose([Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                        Normalize([0.6450, 0.4089, 0.3898], [0.0593, 0.0375, 0.0365])])
    elif data_type == 'NBI':
        return Compose([Resize((image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC),
                        #RandomAffine(degrees=30,translate=(0.3, 0.2)),
                        #RandomRotation(degrees=90),
                        #RandomHorizontalFlip(p=0.5),
                        #RandomVerticalFlip(p=0.5),
                        #RandomAutocontrast(),
                        #ColorJitter(brightness=0.5),
                        ToTensor(),
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


def get_valid_set(dest, classes, data_type, color_dim=3):
    root_dir = myDataset(dest)
    train_dir = join(root_dir, "valid")
    return TrainDatasetFromFolder(train_dir, classes, data_type, color_dim,
                                  input_transform=input_transform(data_type))


def get_test_set(dest, classes, data_type,color_dim=3):
    root_dir = myDataset(dest)
    test_dir = join(root_dir, "valid")
    return TrainDatasetFromFolder(test_dir, classes, data_type, color_dim,
                                 input_transform=test_input_transform(data_type,))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, color_dim=3):
    if color_dim == 1:
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    return img

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
                #path_normal.append(join(class_dir, name_class, pat, 'Normal'))
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
        if self.data_type == 'normal':
            self.image_files = self.image_files_normal
        elif self.data_type == 'NBI':
            self.image_files = self.image_files_NBI
        input = load_img(self.image_files[index], self.color_dim)

        class_name = self.image_files[index].strip().split("\\")[-4]#类型名称
        target = self.classes.index(class_name)
        if self.input_transform:
            input = self.input_transform(input)

        filename = ','.join(self.image_files[index].strip().split("\\")[-4:])

        return input,target,self.image_files[index]

    def __len__(self):
        if self.data_type == 'normal':
            return len(self.image_files_normal)
        elif self.data_type == 'NBI':
            return len(self.image_files_NBI)
        # return len(self.image_files)