
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time
import datetime

from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from compare_methods import *
import utils

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_accuracy = 0
batchsize = 10
num_workers = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])

print('==> Preparing data..')

train_transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.261, 0.261, 0.261], std=[0.134, 0.134, 0.134])
])

val_transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.261, 0.261, 0.261], std=[0.134, 0.134, 0.134])
])


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines if len(line.strip()) > 0]
    return txt_data

class ThyroidDataset(Dataset):
    def __init__(self, root_dir, txt_b, txt_m, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        super(ThyroidDataset, self).__init__()
        self.root_dir = root_dir
        self.txt_path = [txt_b, txt_m]
        self.classes = ['benign', 'malignant']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample


trainset = ThyroidDataset(root_dir='./data', txt_m='./path/m/malignant_train.txt',
                          txt_b='./path/b/benign_train.txt', transform=train_transformer)

valset = ThyroidDataset(root_dir='./data', txt_m='./path/m/malignant_val.txt',
                        txt_b='./path/b/benign_val.txt', transform=val_transformer)

print(trainset.__len__())
print(valset.__len__())


train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False,
                          num_workers=num_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False,
                        num_workers=num_workers, shuffle=False, pin_memory=True)

print('==> Building model..')


# model = VGG('VGG16')
# model = ResNet50()
# model = SENet50()
model = DenseNet121()
modelname = 'VGG16'
# modelname = 'ResNet50'
# model = 'SeNet50'
# model= 'DenseNet121'

model = model.to(device)


if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
scheduler = utils.create_lr_scheduler(optimizer, len(train_loader), epochs=300, warmup=True)

start_time = time.time()
cost = []
epoch_cost = []

def train(epoch):

    print('\nEpoch: %d' % epoch)

    model.train()
    train_loss = 0
    train_correct = 0

    epoch_cost.append(epoch)

    for batch_index, batch_samples in enumerate(train_loader):

        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target.long())
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = output.argmax(dim=1, keepdim=True)
        train_correct += predicted.eq(target.long().view_as(predicted)).sum().item()

    cost.append(train_loss)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("training time {}".format(total_time_str))
    print('Train set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    f = open('result/{}_train.txt'.format(modelname), 'a+')
    f.write('\nTrain set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(np.squeeze(epoch_cost), np.squeeze(cost))
    plt.xlabel('epoch', size=20)
    plt.ylabel('loss', size=20)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title('{} loss curve'.format(modelname), fontsize=20)
    plt.savefig('./result/{}_Loss_curve.png'.format(modelname))


def val(epoch):

    global best_accuracy
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []

        for batch_index, batch_samples in enumerate(val_loader):

            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)

            loss = criterion(output, target.long())
            test_loss += loss.item()

            score = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.long().view_as(predicted)).sum().item()

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, predicted.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

        print('Val set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss/len(val_loader.dataset), correct, len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset)))

        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()

        print('TP =', TP, 'TN =', TN, 'FN =', FN, 'FP =', FP)

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        AUC = roc_auc_score(targetlist, scorelist)

        print('recall: {:.4f}, precision: {:.4f}, F1: {:.4f}, accuracy: {:.4f}, AUC: {:.4f}'.format(r, p, F1, acc, AUC))

        f = open('result/{}_val.txt'.format(modelname), 'a+')
        f.write('\nrecall: {:.4f}, precision: {:.4f}, F1: {:.4f}, accuracy: {:.4f}, AUC: {:.4f}'.format(r, p, F1, acc, AUC))
        f.close()

        if acc > best_accuracy:
            torch.save(model.state_dict(), './result/{}_model.pth'.format(modelname))
            print("Saving FPR, TPR and model")
            best_accuracy = acc


for epoch in range(0, 300):
    train(epoch)
    val(epoch)
    scheduler.step()
