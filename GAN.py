import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 2
image_size = 784
latent_size = 128
hidden_size = 256
batch_size = 16
save_dir = 'fake'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),  
                                     std=(0.5, 0.5, 0.5))])


mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                            batch_size=batch_size,
                                            shuffle=True)


D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
    )

G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
    )

G.to(device)
D.to(device)

G_opt = torch.optim.Adam(G.parameters(), lr=2e-4)
D_opt = torch.optim.Adam(D.parameters(), lr=2e-4)

criterion = nn.BCELoss()

def reset_grad():
    G_opt.zero_grad()
    D_opt.zero_grad()

def denorm(x):
    return ((x+1)/2).clamp(0, 1)

for epoch in range(num_epoch):
    for i, (image, _) in enumerate(data_loader):

        image = image.reshape(batch_size, -1).to(device)
        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        # traing D
        z = torch.randn(batch_size, latent_size).to(device)
        r_out = D(image)
        f_out = D(G(z))

        d_loss = criterion(r_out, real_label) + criterion(f_out, fake_label)

        reset_grad()
        d_loss.backward()
        D_opt.step()

        # train G
        z = torch.randn(batch_size, latent_size).to(device)
        fake_image = G(z)
        f_out = D(fake_image)
        g_loss = criterion(f_out, real_label)


        reset_grad()
        g_loss.backward()
        G_opt.step()

    fake_image = fake_image.reshape(batch_size, 1, 28, 28)
    save_image(denorm(fake_image), os.path.join(save_dir,'fake_{}.png'.format(epoch)))

torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')