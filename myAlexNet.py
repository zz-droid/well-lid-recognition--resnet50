import torchvision.models as models
import torch.nn as nn


alexnet = models.alexnet()


print(alexnet)

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4,padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(64,192,5,1,2)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(192,384,3,1,1)

        self.conv4 = nn.Conv2d(384,256,3,1,1)

        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)

        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.adapool = nn.AdaptiveAvgPool2d(output_size=6)
        ##自适应平均池化，即保持输入的通道数不变，将输出转化为6*6的图形，内部自己计算需要的卷积核大小和stride


        ##最后一步得到的特征图为256*6*6,将其拉直之后维度为256*6*6=9216
        self.fc1 = nn.Linear(9216,4096)

        self.fc2 = nn.Linear(4096,4096)

        self.fc3 = nn.Linear(4096,1000)

    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)


        x = self.conv3(x)
        x = self.relu(x)
        print(x.size())
        x = self.conv4(x)
        x = self.relu(x)
        print(x.size())
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool3(x)
        print(x.size())
        x= self.adapool(x)
        x = x.view(x.size()[0], -1)
        #x = x.view(x.size()[0], -1) 的主要目的是将多维张量展平为二维矩阵
        ##x.view()改变张量形状  x.size()返回张量的维度  因为此时x.size为[4,256,6,6]  x.size()[0]获取的是张量的第0唯，为batch size 数
        ## -1 表示 自动计算该维度的大小,为256*6*6=9216
        ##所以最后得到的矩阵为4*9216的矩阵

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x


import torch

myalexnet = MyAlexNet()

def get_parameter_number(model): ###获得模型的参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


print(get_parameter_number(myalexnet))

img = torch.zeros((4,3,224,224))   ##4为batch size 批处理大小，每4张照片处理一次

out = myalexnet(img)

print(out.size())

