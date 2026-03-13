import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image  # 读取图片数据
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  ##用来显示循环进度
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from model_utils.model import initialize_model


def seed_everything(seed):  ###固定随机种子，保证每次随机出来的结果相同
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


#################################################################
seed_everything(0)
###############################################

HW = 224  ##改变输入图形大小
train_transform = transforms.Compose(  ##用于训练时数据增广
    [
        transforms.ToPILImage(),  ###本来的数据为224*224*3的但是需要将他变为3*224*224形式，用这个可以做到
        transforms.RandomResizedCrop(224),  ##将图片放大然后裁切
        transforms.RandomRotation(50),  ###将图片按随机方式旋转，50代表旋转最大角度为50°
        transforms.ToTensor()  ##将图片变为张量模式
    ]
)
val_transform = transforms.Compose(  ##用于验证时数据增广,验证集的数据不需要放大缩小旋转这种处理
    [
        transforms.ToPILImage(),  ###本来的数据为224*224*3的但是需要将他变为3*224*224形式，用这个可以做到
        # transforms.RandomResizedCrop(224),    ##将图片放大然后裁切
        # transforms.RandomRotation(50),    ###将图片按随机方式旋转，50代表旋转最大角度为50°
        transforms.ToTensor()  ##将图片变为张量模式
    ]
)


class food_Dataset(Dataset):
    def __init__(self, path, mode="train"):  ##传入地址Path 得到数据集X和标签Y
        self.mode=mode
        if mode == "semi":  ##如果是测试模式，那么只用读入数据X，类别Y要靠训练得出
            self.X = self.read_file(path)
        else:
            self.X, self.Y = self.read_file(path)  ##给一个路径，读出来X,Y
            self.Y = torch.LongTensor(self.Y)  ##标签转为长整型
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

    def read_file(self, path):
        if self.mode == "semi":
            file_list = os.listdir(path)  ##列出文件夹下所有的名字

            xi = np.zeros((len(file_list), HW, HW, 3),
                          dtype=np.uint8)  ##创建一个四维0数组，len(file_list)表示有多少张照片，3代表每张照片有R,G,B三个维度

            for j, img_name in enumerate(file_list):  # enumerate()作用是即可以读到下标，也可以读到下标中的内容
                img_name = os.path.join(path, img_name)  ##将上级文件文件如../unlabled/00与其下图片名如0_0合并起来，构成图片路径
                img = Image.open(img_name)
                img = img.resize((HW, HW))
                xi[j, ...] = img  ##把图片放到创建好的数组中
            print("读到了%d个训练数据" % len(xi))
            return xi
        else:
            for i in tqdm(range(5)):  ##看food-11这个文件 有00-10这11个文件夹，说明有11类食物
                file_dir = path + "/%02d" % i  ##保留两位整数挨个读入文件

                file_list = os.listdir(file_dir)  ##列出文件夹下所有的名字

                xi = np.zeros((len(file_list), HW, HW, 3),
                              dtype=np.uint8)  ##创建一个四维0数组，len(file_list)表示有多少张照片，3代表每张照片有R,G,B三个维度
                yi = np.zeros(len(file_list), dtype=np.uint8)
                for j, img_name in enumerate(file_list):  # enumerate()作用是即可以读到下标，也可以读到下标中的内容
                    img_name = os.path.join(file_dir, img_name)  ##将上级文件文件如../labeled/00与其下图片名如0_0合并起来，构成图片路径
                    img = Image.open(img_name)
                    img = img.resize((HW, HW))
                    xi[j, ...] = img  ##把图片放到创建好的数组中
                    yi[j] = i  ##即标签为第i类

                if i == 0:  ###定义一个大数组X，如果是第0类，就指定X为这一类，到后面第1类，第2类....就把他们并到第0类中去
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)  ##在数轴上合并
                    Y = np.concatenate((Y, yi), axis=0)

        print("读到了%d个训练数据" % len(Y))
        return X, Y

    def __getitem__(self, item):
        if self.mode=="semi":
            return self.transform(self.X[item])
        return self.transform(self.X[item]), self.Y[item]  ##返回的是数据增广后的X

    def __len__(self):
        return len(self.Y)


class myModel(nn.Module):
    def __init__(self, numclass):  ##numclass为要分类的个数
        super(myModel, self).__init__()
        ##模型总框架： 3*224*224->512*7*7->拉直->全连接分类
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 卷积 ##输入维度为3，输出维度64，卷积核大小为3，步长为1，padding为1   -->输出为64*224*224
            nn.BatchNorm2d(64),  ##归一化
            nn.ReLU(),
            nn.MaxPool2d(2)  ##此时图形变为64*112*112
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # -->输出为128*112*112
            nn.BatchNorm2d(128),  ##归一化
            nn.ReLU(),
            nn.MaxPool2d(2)  ##此时图形变为128*56*56
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # -->输出为256*56*56
            nn.BatchNorm2d(256),  ##归一化
            nn.ReLU(),
            nn.MaxPool2d(2)  ##此时图形变为256*28*28
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),  ##归一化
            nn.ReLU(),
            nn.MaxPool2d(2)  ##此时图形变为512*14*14
        )

        self.pool2 = nn.MaxPool2d(2)  ##512*7*7

        self.fc1 = nn.Linear(25088, 1000)  # 全连接网络 25088->1000
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, numclass)  ##1000->11

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)  # 拉直
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)
    plt_train_loss = []  ##记录所有轮次的的损失值
    plt_val_loss = []

    plt_train_acc = []  ##记录训练集模型的准确率，比如有一百张照片，预测对了96张，那么准确率就是96%
    plt_val_acc = []

    max_acc = 0.0  ##为了保存最好的模型参数，记录最大的准确率，准确率越大模型效果越好
    for epoch in range(epochs):
        train_loss = 0.0  ##记录每一轮的loss
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        start_time = time.time()  ##记录这一轮开始的时间
        model.train()  ##模型调整为训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)  ##取一组数据放到设备上
            pred = model(x)  ##用模型计算预测值
            train_batch_loss = loss(pred, target)  ##计算每组的loss值
            train_batch_loss.backward()  ##回传
            optimizer.step()  ##计算梯度，更新参数
            optimizer.zero_grad()
            train_loss += train_batch_loss.cpu().item()
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
            ##argmax指出了最大值的下标 pred为预测值，pred.detach().cpu().numpy()将预测值取出来放到cpu上
            ##比如有三张照片，每张照片最后的到的预测结果pred为[[0.1,0.5,0.4],---->[1,      target为[1,
            #                                         [0.2,0.3,0.5],----> 2,               0,
            #                                         [0.2,0.1,0.7]]----> 2]               2]
            # axis=1代表从横轴取最大值,target为标签值，==代表预测值是否与标签值相同,所以按照以上例子，train_acc得到的预测对的个数为2
            # 要计算准确率还要除以总长度

        plt_train_loss.append(train_loss / train_loader.__len__())  ##train_loader.__len__()表示批次,此处记录每一个样本的平均值
        plt_train_acc.append(
            train_acc / train_loader.dataset.__len__())  ##得到准确率  train_loader.dataset.__len__()代表总样本个数_

        model.eval()  ##模型调整为验证模式
        with torch.no_grad():  ##验证集不用积累梯度
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, target)
                val_loss += val_batch_loss.cpu().item()
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
        plt_val_loss.append(val_loss / val_loader.__len__())
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())

        if val_acc > max_acc:
            torch.save(model, save_path)  ##记录损失最小模型到文件中
            max_acc = val_loss  ##更新最小损失值

        print("[%03d/%03d] %2.2f sec(s) Trainloss: %.6f| Valloss:%.6f |Trainacc: %.6f |Valacc:%.6f" % \
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1], plt_train_acc[-1],
               plt_val_acc[-1]))

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc rate")
    plt.legend(["train", "val"])  ##图例
    plt.show()


# train_path = r"D:\第四五节_分类代码\food_classification\food-11\training\labeled"
# val_path=r"D:\第四五节_分类代码\food_classification\food-11\validation"
train_path = r"D:\第四五节_分类代码\food_classification\JPEGImages\training\labeled"
val_path = r"D:\第四五节_分类代码\food_classification\JPEGImages\validation"
no_lable_path=r"D:\第四五节_分类代码\food_classification\JPEGImages\training\unlabeled\00"
train_set = food_Dataset(train_path,"train")
val_set = food_Dataset(val_path,"val")
no_lable_set=food_Dataset(no_lable_path,"semi")
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)  ##取一批数据
val_loader = DataLoader(val_set, batch_size=4, shuffle=True)

# model=myModel(11)
# from torchvision.models import resnet18
# model=resnet18(pretrained=True)    ##pretrain=True代表不仅把大佬的架构搬过来，还把他们的参数值也搬过来，如果是False代表只用架构，参数w从0开始训练
model = initialize_model("resnet50", 5, use_pretrained=True)
in_features = model.fc.in_features  # 提取模型的输入维度  其实就是将卷积拉直之后的那个维度
model.fc = nn.Linear(in_features, 5)  # 最后一层的分类数为5类

lr = 0.001
loss = nn.CrossEntropyLoss()  ##交叉熵损失
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  ##优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "model_save/best_model.pth"
epochs = 15

train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path)

for batch_x, batch_y in train_loader:
    pred = model(batch_x)
