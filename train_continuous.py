from datasets import Dataset, load_dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch_ema
import argparse

import matplotlib.pyplot as plt

from unet import UNet
from bfn_cont import BFNContinuousData
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import resize, convert_to_3_channels, transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True)
    unet = UNet(3, 3).to(device)
    if args.load_model_path is not None:
        unet.load_state_dict(torch.load(args.load_model_path))
        
    bfn = BFNContinuousData(unet, sigma=args.sigma).to(device)

    optim = AdamW(bfn.parameters(), lr=args.lr)
    ema = torch_ema.ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    ema.to(device)

    losses = []
    loss_sum = 0
    for epoch in tqdm(range(1, args.epoch+1), desc='Training', unit='epoch'):
        for X, _ in tqdm(train_loader, desc='Epoch {}'.format(epoch), unit='batch'):
            optim.zero_grad()
            
            loss = bfn.process_infinity(X.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            
            optim.step()
            
            ema.update()
            
            losses.append(loss.item())
            loss_sum = loss.item()
        ave = 0
        for loss in losses:
            ave += loss
        avg_loss = loss_sum / len(losses)
        tqdm.write('Epoch {}: Loss: {:.8f}'.format(epoch, avg_loss))
        if epoch % args.save_every_n_epoch == 0:
            torch.save(unet.state_dict(), f'model_{epoch}.pth')
            tqdm.write('Epoch {}: saved: {}'.format(epoch, f'model_{epoch}.pth'))

    torch.save(unet.state_dict(), args.save_model_path)
    tqdm.write('Epoch {}: saved: {}'.format(epoch, args.save_model_path))


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default="./models/model.pth")
    parser.add_argument("--save_every_n_epoch", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="./animeface")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    
if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    train(args)