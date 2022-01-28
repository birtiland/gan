import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
 
# 定义鉴别器
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # 使用深度卷积网络作为鉴别器
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf), nn.modules.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2), nn.modules.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4), nn.modules.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 8), nn.modules.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, 1), nn.Sigmoid())

        # nn.ReLU
 
    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = self.fc(out.view(-1, 256 * 6 * 6))
        return out
 
 
# 定义生成器
class Generator(nn.Module):
    def __init__(self, nc, ngf, nz, feature_size):
        super(Generator, self).__init__()
        self.prj = nn.Linear(feature_size, nz * 6 * 6)
        # nn.Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 4), nn.modules.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2), nn.modules.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf), nn.modules.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())
 
    def forward(self, x):
        out = self.prj(x).view(-1, 1024, 6, 6)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return out
 
 
# 图片显示
def img_show(inputs, picname):
    plt.ion()
    inputs = inputs / 2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.pause(0.01)
    plt.savefig(picname + ".jpg")
    plt.close()
 
 
# 训练过程
def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=100, print_every=10):
    iter_count = 0
    for epoch in range(epochs):
        for inputs, _ in train_loader:
            real_inputs = inputs # 真实样本
            fake_inputs = g(torch.randn(5, 100)) # 伪造样本
 
            real_labels = torch.ones(real_inputs.size(0)) # 真实标签
            fake_labels = torch.zeros(5) # 伪造标签
 
            real_outputs = d(real_inputs)
            # print("real1 ",real_outputs)
            real_outputs = torch.squeeze(real_outputs, 1)
            # print("real2 ",real_outputs)
            d_loss_real = criterion(real_outputs, real_labels)

            fake_outputs = d(fake_inputs)
            # print(fake_outputs)
            fake_outputs = torch.squeeze(fake_outputs, 1)
            # print(fake_outputs)
            d_loss_fake = criterion(fake_outputs, fake_labels)
 
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
 
            fake_inputs = g(torch.randn(5, 100))
            outputs = d(fake_inputs)
            outputs = torch.squeeze(outputs, 1)
            real_labels = torch.ones(outputs.size(0))
            g_loss = criterion(outputs, real_labels)
 
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
 
            # if (iter_count % show_every == 0):
            #     print('Epoch:{}, Iter:{}, D:{:.4f}, G:{:.4f}'.format(epoch,
            #                                                      iter_count,
            #                                                      d_loss.item(),
            #                                                      g_loss.item()))
            #     picname = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
            #     img_show(torchvision.utils.make_grid(fake_inputs.data), picname)
 
            # if (iter_count % print_every == 0):
            #     print('Epoch:{}, Iter:{}, D:{:.4f}, G:{:.4f}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))

            #     if (iter_count % show_every == 0) :
            #         picname = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
            #         img_show(torchvision.utils.make_grid(fake_inputs.data), picname)

            iter_count += 1
 
            # print('Finished Training！')

        picname = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
        img_show(torchvision.utils.make_grid(fake_inputs.data), picname)
 
 
# 主程序
if __name__ == '__main__':
    # 串联多个变换操作
    data_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(), # 依概率p水平翻转，默认p=0.5
        transforms.ToTensor(), # 转为tensor，并归一化至[0-1]
        # 标准化，把[0-1]变换到[-1,1]，其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。
        # 原来的[0-1]最小值0变成(0-0.5)/0.5=-1，最大值1变成(1-0.5)/0.5=1
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
 
    # 参数data_transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    train_set = datasets.ImageFolder('imgs', data_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=5,
                                               shuffle=True, num_workers=4) # 数据加载
 
    inputs, _ = next(iter(train_loader))
    # make_grid的作用是将若干幅图像拼成一幅图像
    img_show(torchvision.utils.make_grid(inputs), "RealDataSample")
 
    # 初始化鉴别器和生成器
    d = Discriminator(3, 32)
    g = Generator(3, 128, 1024, 100)
 
    criterion = nn.BCELoss() # 损失函数
    lr = 0.0003 # 学习率
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr) # 定义鉴别器的优化器
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr) # 定义生成器的优化器
 
    # 训练
    train(d, g, criterion, d_optimizer, g_optimizer, epochs=10)
