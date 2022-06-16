import hydra
import torch
torch.cuda.empty_cache()
from omegaconf import OmegaConf, DictConfig
import torch.nn as nn
import torch.optim as optim
from torch import device, utils
from torch.utils import data
from torch.utils.data import dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.multiprocessing
import torchvision
import torch.utils.checkpoint
from torchvision import datasets, models, transforms
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelBinarizer
# import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from imutils import paths
from torchvision.io import read_image
import os
from pathlib import Path
import numpy as np
from torchvision.transforms import transforms

from sklearn.metrics import confusion_matrix


# from score_gen.confmatrix import plot_confusion_matrix
#
# from dataset.datasets import Datasets
# import wandb
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

def cross_entropy_loss(output):
    output = torch.max(output, dim=1)
    return output

def label_generator(data):
    labels = {'Crown_and_Root_Rot': 0, 'Healthy Wheat': 1, 'Leaf Rust': 2, 'Wheat Loose Smut': 3}
    for i in range(len(data)):
        data[i] = labels[data[i]]
    return torch.as_tensor(data)
def collate(batch):
    batch = filter(lambda img: img is not None, batch)
    return data.default_collate(list(batch))

def train(config: DictConfig):
    # wandb.init(project=config.wandb_logger.project, entity=config.wandb_logger.entity, group=config.wandb_logger.group)

    cwd = os.getcwd()
    volumes = hydra.utils.instantiate(config.dataset_)
    trainset, testset = volumes.return_total_batches(config.dataset.train_test_split_seed)


    torch.manual_seed(config.global_seed)
    # load_model = config.load_model



    trainloader = DataLoader(dataset=trainset, batch_size=config.training.batch_size, shuffle=True, num_workers=cpu_count()-6, collate_fn= collate)
    testloader = DataLoader(dataset=testset, batch_size=config.training.batch_size, shuffle=True, num_workers=cpu_count()-6 , collate_fn= collate)

    print('done')
    if config.models.model == 'vgg19':
        model = models.vgg19(pretrained=config.pretrain) 
        # print(model.classifier[0].shape)
        model.classifier[3] = nn.Linear(in_features=4096, out_features=512)
        model.classifier[5] = nn.Dropout(config.training.dropout)
        model.classifier[6] = nn.Linear(in_features=512, out_features=config.training.num_classes)
        print(model)

    elif config.models.model == 'Xception':
        model = torchvision.models.Xception(pretrained=config.pretrain)
    
    elif config.models.model == 'InceptionV3':
        model = torchvision.models.InceptionV3(pretrained=config.pretrain)
    else:
        model = torchvision.models.ResNet152(pretrained=config.pretrain)

    model.to(torch.device('cuda:0'))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)

    # Initialize the prediction and label lists(tensors) for confusion matrix
    predlist = torch.zeros(0, dtype=torch.float32)#.to(torch.device('cuda:0'))
    lbllist = torch.zeros(0, dtype=torch.float32)#.to(torch.device('cuda:0'))

    # if load_model:
    #     the_model = torch.load(Path(cwd, 'outputs'))

    for epoch in range(config.training.num_epoch):

        logs = {}
        total_correct = 0
        total_loss = 0
        total_images = 0
        total_val_loss = 0

        if epoch % 5 == 0:
            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            print("Load True: saving checkpoint")
            #torch.save(model.state_dict(), Path(cwd, 'outputs\\cp'))
        #
        # else:
        #     checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
        #                   'optimizer': optimizer.state_dict()}
        #     print("Loade False: saving checkpoint")
        #     save_checkpoint(checkpoint)


        for i, traindata in enumerate(trainloader):
            images = traindata['image'].to(device = 'cuda:0')
            label = label_generator(traindata['label']).to(device = 'cuda:0')
           # print(torch.LongTensor(label))
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            # optim
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(images.float())
            #print(outputs)
            # _, predicted = torch.max(outputs, 1)
            #outputs = cross_entropy_loss(outputs)
            loss = criterion(outputs, label)  # ....>

            # Backward prop
            loss.backward()

            # Updating gradients
            optimizer.step()
            # scheduler.step()

            # Total number of labels
            total_images += label.size(0)

            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)

            # Calculate the number of correct answers
            correct = (predicted == label).sum().item()

            total_correct += correct
            total_loss += loss.item()

            running_trainacc = ((total_correct / total_images) * 100)

            logs['log loss'] = total_loss / total_images
            logs['Accuracy'] = ((total_correct / total_images) * 100)
            # wandb.log({'training accuracy': running_trainacc})
            if i % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, config.training.num_epoch, i + 1, len(trainloader), (total_loss / total_images),
                        (total_correct / total_images) * 100))
            # break

        # Testing the model

    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    
    #         for testdata in testloader:
    #             images = testdata['image']#.to('cuda:0')
    #             label = label_generator(testdata['label'])
    #             labels = label.to(torch.float)#.to('cuda:0')
    #             outputs = model(images.float())
    
    #             _, predicted = torch.max(outputs.data, 1)
    
    #             predlist = torch.cat([predlist, predicted.view(-1)])  # Append batch prediction results
    
    #             lbllist = torch.cat([lbllist, labels.view(-1)])
    
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    
    #             total_losss = loss.item()
    
    #             accuracy = correct / total
    
    #         print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    
    #         logs['val_' + 'log loss'] = total_loss / total
    #         validationloss = total_loss / total
    
    #         validationacc = ((correct / total) * 100)
    #         logs['val_' + 'Accuracy'] = ((correct / total) * 100)
    
    #         # wandb.log({'test accuracy': validationacc, 'val loss': validationloss})
    
    # # # Computing metrics:
    # #
    # conf_mat = confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())
    
    # print(conf_mat)
    # # cls = ["lower grade glioma (LGG)", "Glioblastoma (GBM/high grade glioma)", "Normal Brain"]
    # # # Per-class accuracy
    # class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    # print(class_accuracy)
    # # plt.figure(figsize=(10, 10))
    # # plot_confusion_matrix(conf_mat, cls)
    # # plt.show()
