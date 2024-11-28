# Deep dream
import torch
import torchvision.models as models
from PIL import Image, ImageFilter, ImageChops
import torchvision.transforms as transformer
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg19(pretrained=True)
vgg = vgg.to(device)

modulelist = list(vgg.features.modules())
# preprocess & deprocess

def preprocess(img):
    # Compose，Resize, ToTensor, Normalize
    transformers = transformer.Compose(
        [transformer.Resize((224,224)),
        transformer.ToTensor(),
        transformer.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    img = transformers(img)
    return img


def deprocess(img):
    # 将图像乘上方差和平均 (C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    img = img*std[:,None,None] + std[:,None,None]
    img = img.transpose(0,1) # (H,C,W)
    img = img.transpose(1,2) # (H,W,C)
    return img # (H,W,C)

def prod(image, layer, iterations, lr):
    input_d = preprocess(image).unsqueeze(0) # 预处理图像，添加了一个维度
    input_d = input_d.to(device).requires_grad_(True)
    vgg.zero_grad()
    for i in range(iterations): # 对于每个iter都要走一遍完整的feature过程
        out = input_d
        for j in range(layer):  # 经过layer层，最终得到含有vgg学习到的特征处理的output
            out = modulelist[j+1](out)
        loss = out.norm() # 计算out张量的范数，优化方向要使得loss增大 // 激活之后的激活值越大，说明学到的特征越明显
        loss.backward()
        with torch.no_grad():
            input_d += lr*input_d.grad # loss对input_d的求导， # 为什么是原图像加上梯度  | 梯度的概念 是指变量x增加，y的变化方向 // 画个图就了解了
    with torch.no_grad():
        input_d = input_d.squeeze() # 去除为1的维度，也就是减去batch
        # input_d.transpose_(0,1)
        # input_d.transpose_(1,2) # 将图像格式转为(H,W,C)
        # 使数据限制在01之间
        input_d = np.clip(deprocess(input_d).detach().cpu().numpy(),0,1)
        im = Image.fromarray(np.uint8(input_d*255))
    return im

# vgg_deep_dream 将整个deep_dream的流程转换
def vgg_deep_dream(image, layer, iterations, lr, octave_scale=2, num_octaves=20):

    if num_octaves>0:
        image1 = image.filter(ImageFilter.GaussianBlur(2)) # 对图像进行高斯处理
        if (image1.size[0] / octave_scale < 1 or image1.size[1]/octave_scale<1): # 如果图像缩小<1就不进行处理
            size = image1.size

        else: # 否则缩小图像
            size = (int(image1.size[0]/octave_scale), int(image1.size[1]/octave_scale))
        image1 = image1.resize(size, Image.Resampling.LANCZOS) # 以抗锯齿插值的方法来调整图像大小
        image1 = vgg_deep_dream(image1, layer, iterations, lr, octave_scale, num_octaves-1) # 递归使用
        # 非最后一层的放大结果
        size = (image.size[0], image.size[1]) # 输入的图像大小
        image1 = image1.resize(size, Image.Resampling.LANCZOS) # 将处理后的图像resize为输入图像时候的大小
        image = ImageChops.blend(image, image1, 0.6) # 和图像进行加权融合，

    img_result = prod(image, layer, iterations, lr)
    img_result = img_result.resize(image.size)
    plt.imshow(img_result)
    return img_result

def showDream(path):
    fig, axes = plt.subplots(3,2, figsize = (20,16))

    img = Image.open(path).convert('RGB')
    for i in range(5):
        layer = 2**(i+1)
        deep_dream_im = vgg_deep_dream(img, layer, 6, 0.2)
        axes[i//2,i%2].imshow(deep_dream_im)
        axes[i//2,i%2].set_title(f'vgg_deep_dream_layer{layer}')

showDream('./data/Deep_Dream/night_sky.jpg')