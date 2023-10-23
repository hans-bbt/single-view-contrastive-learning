import timm
import argparse
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights

import myDatasets as md
import csv
import torchvision
from tqdm import tqdm

parser=argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=3, help="")
parser.add_argument("--EPOCHS", type=int, default=100, help="")
parser.add_argument("--lr", type=float, default=0.0001, help="")
parser.add_argument("--cuda", type=str, default="cuda:0", help="")
parser.add_argument("--path_model", type=str, default='store//repalexnet2_heatmap_n100.pth', help="")
parser.add_argument("--path_loss",  type=str, default="store//repalexnet2_heatmap_n100.csv", help="")
opt = parser.parse_args()

classes=['health','lca','mild','prelca','moderate','severe']
traindataset =md.get_training_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
testdataset =md.get_test_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
train_loader=DataLoader(traindataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)
test_loader=DataLoader(testdataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)

import model_repAlexnet
#v=model_repAlexnet.RepAlexNet(num_blocks=[1, 1, 1, 1],num_classes=6).to(opt.cuda)#repalexnet
v=model_repAlexnet.RepAlexNet(num_blocks=[2, 2, 2, 2],num_classes=6).to(opt.cuda)#repalexnet2

criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(v.parameters(), lr=opt.lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)

loss_train_list=[]#保存训练集损失值
acc_test=[]#测试集准确率

for epoch in range(opt.EPOCHS):
    train_loss = 0.0
    train_start_time=time.time()
    accuracy = 0
    total = 0
    for i, (datas, labels,filename) in enumerate(train_loader):
        datas,labels=datas.to(opt.cuda),labels.to(opt.cuda)
        optimizer.zero_grad()
        outputs = v(datas)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)  # 第一个是值的张量，第二个是序号的张量
        total += labels.size(0)
        accuracy += (predicted == labels).sum()
        train_loss += loss.item()


    loss_train_list.append(train_loss)
    print("EPOCHS", epoch + 1, "Training set Loss:{:.5} :".format(train_loss),
          "testing set accuracy:{:.5}".format(float(accuracy.cpu().float() / total) * 100),)
v=model_repAlexnet.repvgg_model_convert(v)
torch.save(v.state_dict(),opt.path_model)

with torch.no_grad():

    accuracy = 0
    total = 0
    for i, (datas, labels,filename) in enumerate(test_loader):
        datas, labels = datas.to(opt.cuda), labels.to(opt.cuda)
        outputs = v(datas)
        val, predicted = torch.max(outputs, dim=1)  # 第一个是值的张量，第二个是序号的张量
        total += labels.size(0)  # labels.size() --> torch.Size([128]), labels.size(0) --> 128
        accuracy += (predicted == labels).sum()  # 相同为1，不同为0，利用sum()求总和

    acc_test.append(float(accuracy.cpu().float() / total) * 100)
    print("测试集准确率", float(accuracy.cpu().float() / total) * 100)

with open(opt.path_loss, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(loss_train_list)
    writer.writerow(acc_test)