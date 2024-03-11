import itertools
from image_pool import ImagePool
from torch.utils.tensorboard import SummaryWriter
from cyclegan import Cycle_Gan_G, Cycle_Gan_D
import argparse
from mydatasets import CreateDatasets
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from utils import train_one_epoch, val


def train(opt):
    batch = opt.batch
    data_path = opt.dataPath
    print_every = opt.every
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = opt.epoch
    img_size = opt.imgsize

    if not os.path.exists(opt.savePath):
        os.mkdir(opt.savePath)

    # 加载数据集
    train_datasets = CreateDatasets(data_path, img_size, mode='train')
    val_datasets = CreateDatasets(data_path, img_size, mode='test')

    train_loader = DataLoader(dataset=train_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                            drop_last=True)

    # 实例化网络
    Cycle_G_A = Cycle_Gan_G().to(device)
    Cycle_D_A = Cycle_Gan_D().to(device)

    Cycle_G_B = Cycle_Gan_G().to(device)
    Cycle_D_B = Cycle_Gan_D().to(device)

    # 定义优化器和损失函数
    optim_G = optim.Adam(itertools.chain(Cycle_G_A.parameters(), Cycle_G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(itertools.chain(Cycle_D_A.parameters(), Cycle_D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    start_epoch = 0
    A_fake_pool = ImagePool(50)
    B_fake_pool = ImagePool(50)

    # 加载预训练权重
    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        Cycle_G_A.load_state_dict(ckpt['Ga_model'], strict=False)
        Cycle_G_B.load_state_dict(ckpt['Gb_model'], strict=False)
        Cycle_D_A.load_state_dict(ckpt['Da_model'], strict=False)
        Cycle_D_B.load_state_dict(ckpt['Db_model'], strict=False)
        start_epoch = ckpt['epoch'] + 1

    writer = SummaryWriter('train_logs')
    # 开始训练
    for epoch in range(start_epoch, epochs):
        loss_mG, loss_mD = train_one_epoch(Ga=Cycle_G_A, Da=Cycle_D_A, Gb=Cycle_G_B, Db=Cycle_D_B,
                                           train_loader=train_loader,
                                           optim_G=optim_G, optim_D=optim_D, writer=writer, loss=loss, device=device,
                                           plot_every=print_every, epoch=epoch, l1_loss=l1_loss,
                                           A_fake_pool=A_fake_pool, B_fake_pool=B_fake_pool)

        writer.add_scalars(main_tag='train_loss', tag_scalar_dict={
            'loss_G': loss_mG,
            'loss_D': loss_mD
        }, global_step=epoch)

        # 保存模型
        torch.save({
            'Ga_model': Cycle_G_A.state_dict(),
            'Gb_model': Cycle_G_B.state_dict(),
            'Da_model': Cycle_D_A.state_dict(),
            'Db_model': Cycle_D_B.state_dict(),
            'epoch': epoch
        }, './weights/cycle_monent2photo.pth')

        # 验证集
        val(Ga=Cycle_G_A, Da=Cycle_D_A, Gb=Cycle_G_B, Db=Cycle_D_B, val_loader=val_loader, loss=loss, l1_loss=l1_loss,
            device=device, epoch=epoch)



def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--imgsize', type=int, default=256)
    parse.add_argument('--dataPath', type=str, default='../monet2photo', help='data root path')
    parse.add_argument('--weight', type=str, default='weights/cycle_monent2photo.pth', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=20, help='plot train result every * iters')
    opt = parse.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)
