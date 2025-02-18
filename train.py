
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time
import datetime
import transforms as T
import utils

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from datasets import ThyroidDataset
from DSMA import DSMANet

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_accuracy = 0
batchsize = 2
num_workers = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])

print('==> Preparing data..')

class SegmentationPresetTrain:
    def __init__(self, mean, std):
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean, std):
        self.transforms = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.261, 0.261, 0.261), std=(0.134, 0.134, 0.134)):

    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


mean = (0.261, 0.261, 0.261)
std = (0.134, 0.134, 0.134)

trainset = ThyroidDataset(root_dir='./data', txt_m='./path/m/malignant_train.txt',
                          txt_b='./path/b/benign_train.txt',
                          transform=get_transform(train=True, mean=mean, std=std))

valset = ThyroidDataset(root_dir='./data', txt_m='./path/m/malignant_val.txt',
                        txt_b='./path/b/benign_val.txt',
                        transform=get_transform(train=False, mean=mean, std=std))


print(trainset.__len__())
print(valset.__len__())

train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False,
                          num_workers=num_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False,
                        num_workers=num_workers, shuffle=False, pin_memory=True)

print('==> Building model..')

model = DSMANet()
modelname = 'DSMA-Net'
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


awl = utils.AutomaticWeightedLoss()
optimizer = optim.Adam([{'params': model.parameters()}, {'params': awl.parameters(), 'weight_decay': 0}])
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

        data, target, mask = batch_samples['img'].to(device), batch_samples['label'].to(device), batch_samples['mask'].to(device)

        optimizer.zero_grad()
        output_cla, output_seg = model(data)

        loss_cla = utils.criterion(output_cla, target.long())
        loss_seg = utils.criterion(output_seg, mask, dice=True)
        loss = awl(loss_cla, loss_seg)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = output_cla.argmax(dim=1, keepdim=True)
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

            data, target, mask = batch_samples['img'].to(device), batch_samples['label'].to(device), batch_samples['mask'].to(device)

            output_cla, output_seg = model(data)

            loss_cla = utils.criterion(output_cla, target.long())
            loss_seg = utils.criterion(output_seg, mask, dice=True)
            loss = awl(loss_cla, loss_seg)
            test_loss += loss.item()

            score = F.softmax(output_cla, dim=1)
            predicted = output_cla.argmax(dim=1, keepdim=True)
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
