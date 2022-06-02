# This script trains and test CNN for classifying Xray images of lungs.
# CNN is an adaptation of AlexNet

# Workaround for libiomp5md.dll error on Windows:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import PV131_dl_utils as pv131dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----Prepare datasets---------------------------------------------------------

# define transformations for images in the training dataset
my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),   
    transforms.RandomHorizontalFlip(), # default probability of flip is 0.5
    transforms.Normalize((0.5), (0.5))
    ])

# set desired batch size for the dataloaders
batch_size = 128

# training data
trainset = torchvision.datasets.ImageFolder(root='./data/chest_xray_256q/train', 
                                            transform=my_transform)
# training data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
# testing data
testset = torchvision.datasets.ImageFolder(root='./data/chest_xray_256q/test', 
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Grayscale(),
                                                transforms.Normalize((0.5),
                                                                     (0.5))]))
# testing data loader
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
# define the class names
classes = ('normal', 'pneumonia bacteria', 'pneumonia virus')

#-----Neural network-----------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()       
        self.features1 = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),           
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=96),
            )

        self.features2 = nn.Sequential(  
            # layer 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),
            )

        self.features3 = nn.Sequential(  
            # layer3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),            
            # layer 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),            
            # layer 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            # layer 6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # layer 7
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            # layer 8
            
            nn.Linear(1024, 3),
        )

    def forward(self, x):   
        x = self.features1(x)
        torch._assert(x.shape[1:]==torch.Size([96, 30, 30]),'incorrect shape')
        x = self.features2(x)
        torch._assert(x.shape[1:]==torch.Size([256, 13, 13]),'incorrect shape')
        x = self.features3(x)
        torch._assert(x.shape[1:]==torch.Size([256, 6, 6]),'incorrect shape')
        x = self.classifier(x)
        torch._assert(x.shape[1:]==torch.Size([3]),'incorrect shape')
        return x   

# allocate network in the GPU / CPU memory
net = Net().to(device)

# print network summary
summary(net, trainset[0][0].shape);

# define optimizer and learning rate
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# define loss function
loss_fn = F.cross_entropy


#-----Training my network------------------------------------------------------

# optional - reset weights
'''net.apply(pv131dl.reset_weights);'''

# train the network
# (10 epochs require approximately 10 minutes of training on a GPU)
tr_l, tr_c = pv131dl.train(net, batch_size, 10, trainloader,
                           loss_fn, optimizer, device)

# save the trained model
torch.save(net.state_dict(), './my_network.pth')

# plot training loss
'''fig = plt.figure()
plt.plot(tr_c, tr_l, color='blue')
plt.xlabel('number of training examples seen')
plt.ylabel('cross entropy loss')'''

# load the trained model
'''net = Net()
net.load_state_dict(torch.load('./my_network.pth'))
net.to(device)'''

#-----Testing the Trained Network----------------------------------------------

# get one testing batch and make predictions
example_data, example_targets, predictions = pv131dl.visual_test(net,
                                                                 testloader,
                                                                 device)

#-----Evaluating the Network---------------------------------------------------
lab_pred, lab_targ = pv131dl.test(net, batch_size, testloader, device);

# Compute and display the confusion matrix
conf_matrix = pv131dl.compute_confusion_matrix(lab_targ, lab_pred)
pv131dl.show_confusion_matrix(conf_matrix, classes)
  
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print('\nAccuracy {:.1f}%\n'.format(accuracy*100))

