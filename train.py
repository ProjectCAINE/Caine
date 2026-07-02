import torch
import torch.nn as nn
import time
import os
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import VisionTransformer, Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

model = VisionTransformer(n_embd=512, n_head=8, n_layer=8).to(device)
decoder = Decoder(n_embd=512).to(device)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
decoder.apply(init_weights)

if os.path.exists('caine.pth'):
    model.load_state_dict(torch.load('caine.pth'))

dataset = ImageFolder(root='caine_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-5, max_lr=1e-3,
    step_size_up=500, mode="triangular"
)

try:
    print("Starting Training with Cyclic LR")
    while True:
        for images, _ in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            latent_thought = model(images)
            latent_thought = latent_thought + 0.05 * torch.randn_like(latent_thought)
            reconstruction = decoder(latent_thought)

            loss = F.mse_loss(reconstruction, images) + 0.00001 * (
                torch.sum(torch.abs(reconstruction[:, :, :, :-1] - reconstruction[:, :, :, 1:])) +
                torch.sum(torch.abs(reconstruction[:, :, :-1, :] - reconstruction[:, :, 1:, :]))
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.save(reconstruction, 'caine_vision.pt.tmp')
            os.replace('caine_vision.pt.tmp', 'caine_vision.pt')

            if time.time() % 60 < 1:
                torch.save(model.state_dict(), 'caine.pth')
                print(f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            time.sleep(0.05)
except KeyboardInterrupt:
    torch.save(model.state_dict(), 'caine.pth')
    print("Saved training data")
