import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import os

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import transforms as T
from datasets import ThyroidDataset
from DSMA import DSMANet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batchsize = 10
num_workers = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])

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


def plot_confusion_matrix(cm,
                          target_names,
                          modelname,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see:
                  http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm = cm,
                           normalize = True, # show proportions
                          target_names = y_labels_vals, # list of classes names
                          title = best_estimator_name) # title of graph
    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.rcParams['font.size'] = 20
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, size=20)
        plt.yticks(tick_marks, target_names, size=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=20)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.savefig('./result/{}.jpg'.format(modelname), bbox_inches='tight')
    print('Saving Confusion Matrix..')
    plt.show()

mean = (0.261, 0.261, 0.261)
std = (0.134, 0.134, 0.134)

testset = ThyroidDataset(root_dir='./data/', txt_m='./path/m/malignant_test.txt',
                         txt_b='./path/b/benign_test.txt', transform=get_transform(train=False, mean=mean, std=std))

print(testset.__len__())

test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False,
                         num_workers=num_workers, shuffle=False, pin_memory=True)

def test():


    model = DSMANet()
    modelname = 'DSMA-Net'

    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.load_state_dict(torch.load('./result/{}_model.pth'.format(modelname)))
    model.eval()

    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []

        for batch_index, batch_samples in enumerate(test_loader):

            data, target, mask = batch_samples['img'].to(device), batch_samples['label'].to(device), batch_samples['mask'].to(device)

            output_cla, output_seg = model(data)

            score = F.softmax(output_cla, dim=1)
            pred = output_cla.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()

        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP', TP + FP)

        p = TP / (TP + FP)
        print('precision', p)

        r = TP / (TP + FN)
        print('recall', r)

        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1', F1)
        print('acc', acc)

        AUC = roc_auc_score(targetlist, scorelist)
        print('AUC', AUC)

        C_M = confusion_matrix(targetlist, predlist)
        print('confusion matrix', C_M)
        plot_confusion_matrix(C_M, target_names=['Malignant', 'Benign'],
                              modelname=modelname, title='Confusion matrix', normalize=False)

        fpr, tpr, _ = roc_curve(targetlist, scorelist)
        with open('./result/{}_roc.pkl'.format(modelname), "wb") as fw:
            pickle.dump((fpr, tpr, AUC), fw)
        with open('./result/{}_roc.pkl'.format(modelname), "rb") as fr:
            FPR, TPR, AUC = pickle.load(fr)

        plt.figure(figsize=(8, 6))
        plt.plot(FPR, TPR, color='red', linewidth=2, label='{}_AUC = {:.4f}'.format(modelname, AUC))
        plt.xlabel('False Positive Rate', size=20)
        plt.ylabel('True Positive Rate', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title('ROC curve', fontsize=20)
        plt.legend(loc="lower right", prop={'size': 16})
        plt.savefig('./result/{}_ROC_curve.png'. format(modelname))

test()
