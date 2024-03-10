#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from net import Generator
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator().to(device)

## 载入模型权重
modelpath = sys.argv[1] 
savepath = sys.argv[2] 
netG.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))
netG.eval() ## 设置推理模式，使得dropout和batchnorm等网络层在train和val模式间切换
torch.no_grad() ## 停止autograd模块的工作，以起到加速和节省显存
nz = 100

for i in range(0,100):
    noise = torch.randn(64, nz, 1, 1, device=device)
    fake = netG(noise).detach().cpu()
    rows = vutils.make_grid(fake, padding=2, normalize=True)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(rows, (1, 2, 0)))
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(os.path.join(savepath,"%d.png" % (i)))
    plt.close(fig)

