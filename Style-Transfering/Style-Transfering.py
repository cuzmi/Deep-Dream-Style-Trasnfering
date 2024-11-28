import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transformer
import torchvision.models as models

import copy
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_avaiable() else 'cpu')
style_path = '##.jpg'
content_path = '##.jpg'

# 内容损失 ~ 用和原来图像的mse来衡量

class ContentLoss(nn.Module):
    def __init__(self, target):  # 这里的target为原始的Content内容，且一直保留到后面
        super().__init__()
        # target为tensor
        self.target = target.detach()  # detach是为了分离出计算图

    def forward(self, img):
        self.loss = F.mse_loss(img, self.target)
        return img  # ? 为什么返回的是img // forward返回的是网路的输出 // 但是loss的输出不就应该是loss吗


# 风格损失 ~ 采用的是底层的一些卷积层的子集 // 方便学习到泛化的，不那么抽象的特征

# 采用格拉姆矩阵来计算
def gram_matrix(img):  # ? 为什么格拉姆能计算出风格？ 看起来像是自相关性
    a, b, c, d = img.size()  # 得到 batch_size, channel, h, w

    features = img.view(a * b, c * d)  # 计算图总量 * 计算图内部元素量

    G = torch.mm(features, features.t())  # mm矩阵相乘 feature和feature的转置
    return G.div(a * b * c * d)  # 除以abcd归一化 // 防止以外的因素对数值产生影响


class StyleLoss(nn.Module):
    def __init__(self, target_feature):  # 这个是目标style的feature
        super().__init__()
        self.target = gram_matrix(target_feature).detach()  # (C,C)

    def forward(self, img):
        G = gram_matrix(img)  # 训练对象的style (C,C)
        self.loss = F.mse_loss(G, self.target)
        return img

    # 导入数据进行处理

transformers = transformer.Compose(
    [transformer.Resize((512, 600)),
     transformer.ToTensor()]  # 没有进行Normalize，在模型中进行Normalize
)


def image_loader(path):
    image = Image.open(path)
    image = transformers(image).unsqueeze(0)  # 加batch
    return image.to(device, torch.float)


style_img = image_loader(style_path)
content_img = image_loader(content_path)

# 显示图像
unloader = transformer.ToPILImage()

plt.ion()  # 启用交互模式，动态更新图像


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.01)


# plt.figure()
# imshow(style_img, title='Style_Img')
# imshow(content_img, title='Content_Img')

# 下载预训练模型
net = models.vgg19(pretrained=True).features.to(device).eval()  # 原来模型可以拆开来to device的啊
# print(net)

# 选择优化器
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# 构建模型
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)  # mean (1,C,1,1) 利于广播
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)  # (1,C,1,1)

    def forward(self, x):
        return x - self.mean / self.std


def get_style_model_and_losses(net, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    net = copy.deepcopy(net)

    # 标准化模型
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 初始化损失值
    content_losses = []
    style_losses = []

    # 使用Sequential方法构建模型
    model = nn.Sequential(normalization)

    i = 0
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise Runtime('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)  # 原本的net - (9):nn.Conv2d ~ 而name则是更换了前面的index 变成了 (conv2d):nn.Conv2d

        # 整个是模型是动态增加的，且到一个点就进行计算
        if name in content_layers:  # 前面提到过，浅层能很好地进行重建工作能提取到具体明显的内容，因此计算内容的loss就用前几层的feature结果来计算
            # 累加内容损失
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)


        if name in style_layers:  # 整个layer是进行集合组合的 : 1->1+2->1+2+3+4->··
            # 累加风格损失
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    # 对内容损失和风格损失之后的层进行修剪
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


# 训练模型
def run_style_transfer(net, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=600,
                       style_weight=100000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(net, normalization_mean, normalization_std,
                                                                     style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():  # 函数居然还能传递给优化器啊
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print('run {}:'.format(run))
                print('Style loss : {:4f} Content loss : {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

def imshow(tensor):
    if tensor.dim == 4:
        tensor = tensor.squeeze(0)
    img = transformer.ToPILImage()(tensor)
    plt.imshow(img)

final = run_style_transfer(net,normalization_mean=0.5,normalization_std=0.5,content_img,style_img,content_img)
imshow(final)