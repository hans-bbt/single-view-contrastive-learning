import argparse
import torch
from torch.utils.data import DataLoader
import myDatasets as md
from model_selfcontrast_alexnet import contrast_model
import torch.nn.functional as F


parser=argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=1, help="")
parser.add_argument("--cuda", type=str, default="cuda:0", help="")
opt = parser.parse_args()

classes=['health','lca','mild','prelca','moderate','severe']
traindataset =md.get_training_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
testdataset =md.get_test_set(md.myDataset("F:/laryngoscopic/class6_new_"),classes,"NBI")
train_loader=DataLoader(traindataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)
test_loader=DataLoader(testdataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)

model=contrast_model().to(opt.cuda)
model.load_state_dict(torch.load('store\\pth_\\self_contrast_loss_maxandrand.pth'))

with torch.no_grad():
    accuracy = 0
    total = 0
    model.eval()
    for i, (datas, labels,filename) in enumerate(test_loader):
        datas, labels = datas.to(opt.cuda), labels.to(opt.cuda)
        data_224 = F.interpolate(datas, size=(224, 224), mode='bilinear', align_corners=False)
        outputs = model(data_224,"test")
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        if predicted != labels:
            print("实际标签", filename[0].strip().split("\\")[-4],end="")
            print("预测", classes[predicted])
        accuracy += (predicted == labels).sum()
    print("测试集准确率", float(accuracy.cpu().float() / total) * 100)
    print(total)