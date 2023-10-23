#compare with max1 and max2
import argparse
import time
import torch
from torch import optim
from torch.utils.data import DataLoader

import myDatasets as md
import csv
from model_selfcontrast_alexnet import contrast_model
from loss.loss_selfcontrast0 import loss_selfcontrast
import torch.nn.functional as F
from tqdm import tqdm

parser=argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=40, help="")
parser.add_argument("--EPOCHS", type=int, default=500, help="")
parser.add_argument("--lr", type=float, default=0.0001, help="")
parser.add_argument("--cuda", type=str, default="cuda:0", help="")
parser.add_argument("--path_model", type=str, default='store\\pth_\\self_contrast_maxrand_resnet18.pth', help="")
parser.add_argument("--path_loss",  type=str, default="store\\loss_csv\\self_contrast_max2.csv", help="")
parser.add_argument("--path_false", type=str, default="store\\false_info\\self_contrast_max2.csv", help="")
opt = parser.parse_args()

classes=['health','lca','mild','prelca','moderate','severe']
traindataset =md.get_training_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
testdataset =md.get_test_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
train_loader=DataLoader(traindataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)
test_loader=DataLoader(testdataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)

model=contrast_model().to(opt.cuda)

criterion = loss_selfcontrast()
optimizer=optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
loss_list=[]
acc_list=[]
location_list=[]

with open("gradcam\\localinfo\\local_info_max2.csv",'r',encoding="utf-8") as file_obj:
    reaader=csv.reader(file_obj)
    for r in reaader:
        if len(r)!=0:
            location_list.append(r)

for epoch in range(opt.EPOCHS):
    train_loss = 0.0
    accuracy = 0
    total = 0
    model.train()
    loop = tqdm((train_loader), total=len(train_loader))
    for datas, labels,filename in loop:
        datas,labels=datas.to(opt.cuda),labels.to(opt.cuda)
        loop.set_description(f'Epoch [{epoch}/{opt.EPOCHS}]')
        optimizer.zero_grad()

        data_224 = F.interpolate(datas, size=(224, 224), mode='bilinear', align_corners=False)
        max_regions=[]
        for x in filename:
            result = [sublist for sublist in location_list if sublist[0] == x]
            max_regions.append([eval(result[0][1]), eval(result[0][2])],)

        crop_size = int(1080 * 0.4)  # 计算剪裁尺寸
        cropped_tensors1 = []
        cropped_tensors2 = []

        for i,coords in  enumerate(max_regions):

            top = int(coords[0][0]*1080)
            left = int(coords[0][1]*1080)
            bottom = top + crop_size
            right = left + crop_size
            cropped_x = datas[i, :, top:bottom, left:right]
            cropped_x = F.interpolate(torch.unsqueeze(cropped_x, dim=0), size=(224, 224), mode='bilinear', align_corners=False)
            cropped_tensors1.append(cropped_x)

            top = int(coords[1][0]*1080)
            left = int(coords[1][1]*1080)
            bottom = top + crop_size
            right = left + crop_size
            cropped_x = datas[i, :, top:bottom, left:right]
            cropped_x = F.interpolate(torch.unsqueeze(cropped_x, dim=0), size=(224, 224), mode='bilinear', align_corners=False)
            cropped_tensors2.append(cropped_x)

        cropped_tensors1 = torch.cat(cropped_tensors1, dim=0).to(opt.cuda)
        cropped_tensors2 = torch.cat(cropped_tensors2, dim=0).to(opt.cuda)

        outputs,outputs_crop1,outputs_crop2 = model(data_224,cropped_tensors1,cropped_tensors2,"train")
        loss=criterion(outputs,outputs_crop1,outputs_crop2,labels)

        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)  # 第一个是值的张量，第二个是序号的张量
        total += labels.size(0)
        accuracy += (predicted == labels).sum()
        train_loss += loss.item()
        loop.set_postfix(loss=train_loss, acc=(float(accuracy.cpu().float() / traindataset.__len__()) * 100))
    loss_list.append(train_loss)
torch.save(model.state_dict(),opt.path_model)

for i in range(10):
    with torch.no_grad():
        accuracy = 0
        total = 0
        model.eval()
        for i, (datas, labels,filename) in enumerate(test_loader):
            datas, labels = datas.to(opt.cuda), labels.to(opt.cuda)
            data_224 = F.interpolate(datas, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = model(data_224,_,_,"test")
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
        acc_list.append(float(accuracy.cpu().float() / total) * 100)
        print("测试集准确率", float(accuracy.cpu().float() / total) * 100)

with open(opt.path_loss, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(loss_list)
    writer.writerow(acc_list)