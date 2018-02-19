# _*_ coding:utf-8 _*_
import torch
import torchvision as tv
from model import Generator, Discriminator
from torch.autograd import Variable
import torch.nn as nn


if __name__ == '__main__':
    BATCH_SIZE = 128
    IMAGESIZE = 96
    EPOCH = 250
    N_noise = 100

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(IMAGESIZE),
        tv.transforms.CenterCrop(IMAGESIZE),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder('data/', transform=transforms)
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=2,
                                              drop_last=True
                                              )

    D = Discriminator()
    G = Generator()
    D = D.cuda()
    G = G.cuda()

    loss_func = nn.BCELoss()
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


    for epoch in range(EPOCH):
        for step, (images, _) in enumerate(trainloader):
            D.zero_grad()
            real_images = Variable(images.cuda())
            d_real = D(real_images)
            d_realloss = loss_func(d_real, Variable(torch.ones(BATCH_SIZE).cuda()))
            d_realloss.backward()

            G_ideas = Variable(torch.randn(BATCH_SIZE, N_noise, 1, 1).cuda())
            G_fakeimage = G(G_ideas)
            d_fake = D(G_fakeimage.detach())
            d_fakeloss = loss_func(d_fake, Variable(torch.zeros(BATCH_SIZE).cuda()))
            d_fakeloss.backward()
            d_loss = d_realloss + d_fakeloss
            optimizerD.step()

            if step % 5 == 0:
                G.zero_grad()
                out = D(G_fakeimage)
                g_loss = loss_func(out, Variable(torch.ones(BATCH_SIZE).cuda()))
                g_loss.backward()
                optimizerG.step()
            if step % 50 == 0:
                tv.utils.save_image(G_fakeimage.data[:64].cpu(), '%s/%s_%s.png' % ('imgs/', epoch, step),
                                    normalize=True,
                                    range=(-1, 1))
        if epoch % 50 == 0:
            torch.save(D.state_dict(), 'checkpoints/d_%s.pth' % epoch)
            torch.save(G.state_dict(), 'checkpoints/g_%s.pth' % epoch)

    torch.save(G.state_dict(), "anime_face_g.pkl")
    torch.save(D.state_dict(), "anime_face_d.pkl")
