import torch
import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from utils import convert_to_3_channels, resize
from unet import UNet
from bfn_cont import BFNContinuousData
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.transforms as transforms

import torch_ema

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_output_image(args, bfn):
    # Comparison of Input and Output images    
    transform = transforms.Compose([
        resize(h=args.height, w=args.width),
        convert_to_3_channels,
        transforms.ToTensor(),  # Convert images to tensors
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the pixel values to [-1, 1]
    ])
    dataset = datasets.ImageFolder(root="./animeface_valid", transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    for batch in data_loader:
        x, _ = batch
        break
    
    mus = [[] for _ in range(args.batch)]
    preds = [[] for _ in range(args.batch)]
    ts = []
    with torch.no_grad():
        for t in np.linspace(0.01, 1, 20):
            _, mu, pred, _ = bfn.process_infinity(x=x.to(device), t=torch.full((args.batch, ), t, device=device), train=False)
            for b in range(args.batch):
                mus[b].append(mu.permute(0, 2, 3, 1).cpu().detach().numpy()[b])
                preds[b].append(pred.permute(0, 2, 3, 1).cpu().detach().numpy()[b])
            ts.append(t)
        
    fig, ax = plt.subplots(2*args.batch, 20, figsize=(20, 2*args.batch))

    for b in range(args.batch):
        for i in range(20):
            ax[2*b, i].imshow(mus[b][i])
            ax[2*b+1, i].imshow(preds[b][i])
            ax[2*b, i].axis('off')
            ax[2*b+1, i].axis('off')
    for i in range(20):
        ax[0, i].set_title(np.round(ts[i], 3))
    
    plt.savefig('./outputs/validation_image_plots.png')
    

def generate(args):
    unet = UNet(3, 3).to(device=device)
    unet.load_state_dict(torch.load(args.load_model_path))
    unet.eval()
    bfn = BFNContinuousData(unet, in_channels=3, sigma=0.01).to(device)
    bfn.eval()
    ema = torch_ema.ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    ema.to(device)
    return_samples = args.step//10
    with torch.no_grad():
        x_hat, out_list = bfn.sample(args.height, args.width, device=device, steps=args.step, return_samples=return_samples, batch_size=args.batch, ema=None)
    
    fig, ax = plt.subplots(args.batch, len(out_list), figsize=(10, args.batch))
    for i in range(len(out_list)):
        for b in range(len(out_list[i])):
            ax[b, i].imshow(((out_list[i][b]+1)/2*255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy())
            ax[b, i].axis('off')
    for i in range(len(out_list)):
        ax[0, i].set_title(np.round(return_samples*i, 2))
    plt.savefig('./outputs/generate_plots.png') # Generated image every step//10
    
    # Save PIL Image
    to_pil = ToPILImage()
    pil_images = [to_pil(((image+1)/2*255).clamp(0, 255).to(torch.uint8)) for image in x_hat]
    for i, pil_image in enumerate(pil_images):
        pil_image.convert("RGB").save(f'./outputs/image_{i}.png') # Generated images
        
    # Comparison of Input and Output images
    if args.generate_input_output_image:
        input_output_image(args, bfn)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model_path", type=str, required=True, default="./models/model.pth")
    parser.add_argument("--data_path", type=str, default="./animeface")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generate_input_output_image", type=bool, default=False)
    return parser
    
if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    generate(args)