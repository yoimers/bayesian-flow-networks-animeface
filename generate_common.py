

import argparse
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch_ema
from bfns.bfn_continuous import BFNContinuousData
from bfns.bfn_discretised import BFNDiscretisedData
from torchvision.transforms import ToPILImage
from matplotlib.animation import FuncAnimation

from networks.unet import UNet
from train_common import BFNType, TimeType
from utils import default_transform, set_seed
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_output_image(args, bfn: BFNContinuousData, bfnType: BFNType, timeType: TimeType):
    # Comparison of Input and Output images    
    transform = default_transform(args.height, args.width)
    dataset = datasets.ImageFolder(root="./animeface_valid", transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    x, _ = next(iter(data_loader))

    mus = [[] for _ in range(args.batch)]
    preds = [[] for _ in range(args.batch)]
    ts = []
    col = 20
    with torch.no_grad():
        for t in np.linspace(0.001, 0.66, col):
            if timeType == TimeType.ContinuousTimeLoss:
                time = torch.full((len(x), ), t, device=device)
                _, mu, pred, _ = bfn.process_infinity(x=x.to(device), t=time, return_params=True)
            elif timeType == TimeType.DiscreteTimeLoss:
                step = torch.full((len(x), ), int(args.step*t), device=device)
                _, mu, pred, _ = bfn.process_discrete(x=x.to(device), step=step, max_step=args.step, return_params=True)
            else:
                raise ValueError("The timeType must be either ContinuousTimeLoss or DiscreteTimeLoss.")
            
            for b in range(len(x)):
                mus[b].append(mu.permute(0, 2, 3, 1).cpu().detach().numpy()[b])
                preds[b].append(pred.permute(0, 2, 3, 1).cpu().detach().numpy()[b])
            ts.append(t)

    fig, ax = plt.subplots(2 * args.batch, col, figsize=(12, args.batch))

    for b in range(args.batch):
        for i in range(col):
            ax[2*b, i].imshow(mus[b][i])
            ax[2*b+1, i].imshow(preds[b][i])
            ax[2*b, i].axis('off')
            ax[2*b+1, i].axis('off')
    for i in range(col):
        ax[0, i].set_title(np.round(ts[i], 3))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
    plt.savefig('./outputs/validation_image_plots.png')

def common_generate(args: argparse.ArgumentParser, bfnType: BFNType, timeType: TimeType):
    set_seed(args.seed)
    if bfnType == BFNType.Continuous:
        unet = UNet(3, 3).to(device=device)
    elif bfnType == BFNType.Discretized:
        unet = UNet(3, 6).to(device=device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    unet.load_state_dict(torch.load(args.load_model_path))
    unet.eval()
    
    if bfnType == BFNType.Continuous:
        bfn = BFNContinuousData(unet, in_channels=3, sigma=args.sigma).to(device)
    elif bfnType == BFNType.Discretized:
        bfn = BFNDiscretisedData(unet, K=args.K, in_channels=3, sigma=args.sigma).to(device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    bfn.eval()
    
    ema = torch_ema.ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    ema.to(device)
    
    if args.gif_generate:
        return_samples = args.gif_save_every_n_step
    else:
        return_samples = args.step//10
        
    with torch.no_grad():
        x_hat, out_list = bfn.sample(args.height, args.width, device=device, steps=args.step, return_samples=return_samples, batch_size=args.batch, ema=ema)
    
    if args.gif_generate:
        fig, ax = plt.subplots(1, args.batch, figsize=(2 * args.batch, 2))
        def animate(frame):
            for i in range(args.batch):
                ax[i].clear()
                ax[i].imshow(((out_list[frame][i]+1)/2*255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy())
                ax[i].axis('off')
                ax[i].set_title(f'Step {max(1, args.gif_save_every_n_step * frame)}', fontsize=12)
                
        anim = FuncAnimation(fig, animate, frames=args.step//args.gif_save_every_n_step, interval=100)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
        anim.save("./outputs/generate_output.gif", writer='pillow')
    else:
        fig, ax = plt.subplots(args.batch, len(out_list), figsize=(8, args.batch))
        for i in range(len(out_list)):
            for b in range(len(out_list[i])):
                ax[b, i].imshow(((out_list[i][b]+1)/2*255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy())
                ax[b, i].axis('off')
        for i in range(len(out_list)):
            ax[0, i].set_title(np.round(return_samples*i, 2))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        plt.savefig('./outputs/generate_plots.png') # Generated image every step//10
        
    # Save PIL Image
    to_pil = ToPILImage()
    pil_images = [to_pil(((image+1)/2*255).clamp(0, 255).to(torch.uint8)) for image in x_hat]
    for i, pil_image in enumerate(pil_images):
        pil_image.convert("RGB").save(f'./outputs/image_{i}.png') # Generated images
        
    # Comparison of Input and Output images
    if args.generate_input_output_image:
        input_output_image(args, bfn, bfnType=bfnType, timeType=timeType)

def setup_generate_common_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--load_model_path", type=str, required=True, default="./models/model.pth", help="Path to the pre-trained model file.")
    parser.add_argument("--data_path", type=str, default="./animeface", help="Path to the dataset directory.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--step", type=int, default=1000, help="Number of generation steps.")
    parser.add_argument("--sigma", type=float, default=0.001, help="Value of sigma for generation.")
    parser.add_argument("--height", type=int, default=32, help="Height of the generated images.")
    parser.add_argument("--width", type=int, default=32, help="Width of the generated images.")
    parser.add_argument("--K", type=int, default=16, help="Number of Bins.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generate_input_output_image", action="store_true", help="Whether to generate input(θ, mu, ...) and output images.")
    parser.add_argument("-gif", "--gif_generate", action="store_true", help="Whether to generate output images gif.")
    parser.add_argument("--gif_save_every_n_step", type=int, default=50)
    return parser