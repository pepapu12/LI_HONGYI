import torch
import torchvision
import torchvision.transforms as transforms
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset =  torchvision.datasets.CIFAR10(root="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data", train=True,download=True,transform=transform)
trainloader= torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
testset =  torchvision.datasets.CIFAR10(root="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data", train=True,download=True,transform=transform)
testloader= torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)

classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter= iter(trainloader)
images,labels=next(dataiter)

print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

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
        #变化x的形状以适配全连接层的输入
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

net=Net()
net.to(device)
# print(net)
#Cross Entropy Loss and gradiant descent optimization
import torch.optim as optim

#定义损失函数，选用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)#学习率0.001

#在训练集上训练模型, 需要多次迭代

for epoch in range(5):
    running_loss= 0.0

    for i,data in enumerate(trainloader,0):
        inputs,labels=data[0].to(device),data[1].to(device)
        optimizer.zero_grad() #to clean grad from last iteration

        outputs=net(inputs)

        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if(i+1)%2000==0:
            print('[%d,%5d] loss :%.3f'%(epoch +1,i+1,running_loss/2000))
            running_loss=0.0

print('Finished Training')
PATH="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data\\cifar10_net.pth"
torch.save(net.state_dict(),PATH)
