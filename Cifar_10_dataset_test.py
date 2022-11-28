import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

testset =  torchvision.datasets.CIFAR10(root="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data", train=True,download=True,transform=transform)

testloader= torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)
dataiter=iter(testloader)
images,labels=next(dataiter)

classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1= nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,32)
        self.fc4=nn.Linear(32,10)

    def forward(self,x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        #Change x to be capable with full connection
        x=x.view(-1,16*5*5)
        #1st hidden layer with linear transformation and relu funciton
        x=F.relu(self.fc1(x))
        #2nd hidden layer with linear transformation and relu funciton
        x=F.relu(self.fc2(x))
        #3rd hidden layer with linear transformation and relu funciton
        x=F.relu(self.fc3(x))
        #4th hidden layer with no relu function
        x=self.fc4(x)
        return x
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=Net()
net.to(device)
PATH="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data\\cifar10_net.pth"
net.load_state_dict(torch.load(PATH,map_location='cpu'))

correct=0
total=0
class_correct = list(0. for i in range(10))
class_total=list(0. for i in range (10))
with torch.no_grad():
    for data in testloader:
        images,labels=data
        outputs=net(images)
        _, predicted=torch.max(outputs,1)
        c=(predicted==labels).squeeze()
        for i in range(4):
            label=labels[i]
            class_correct[label]+=c[i].item()
            class_total[label]+=1
        total+=labels.size(0)
        correct +=(predicted==labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d%%'%(100*correct/total))
for i in range(10):
    print('Accuracy of %5s : %2d %%'%(classes[i],100*class_correct[i]/class_total[i]))