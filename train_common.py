from datasets import Dataset, load_dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
import torch_ema
import argparse

import matplotlib.pyplot as plt

import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from bfns.bfn_continuous import BFNContinuousData
from bfns.bfn_discretised import BFNDiscretisedData
from networks.unet import UNet
from utils import default_transform
import argparse
from enum import Enum, auto
from torch.utils.tensorboard import SummaryWriter
from data.molecule_dataset import MoleculeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BFNType(Enum):
    Continuous = auto()
    Discretized = auto()

class TimeType(Enum):
    ContinuousTimeLoss = auto()
    DiscreteTimeLoss = auto()

def train(args: argparse.ArgumentParser, bfnType: BFNType, timeType: TimeType):
    dataset = MoleculeDataset("bfn_exercise.pkl", conditioned=args.conditioned)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, num_workers=8)
    
    if bfnType == BFNType.Continuous:
        unet = UNet(1, 1).to(device)
    elif bfnType == BFNType.Discretized:
        unet = UNet(3, 6).to(device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    
    if args.load_model_path is not None:
        unet.load_state_dict(torch.load(args.load_model_path))
        
    if bfnType == BFNType.Continuous:
        bfn = BFNContinuousData(unet, sigma=args.sigma).to(device)
    elif bfnType == BFNType.Discretized:
        bfn = BFNDiscretisedData(unet, K=args.K, sigma=args.sigma).to(device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    
    optimizer = AdamW(unet.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    ema = torch_ema.ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    ema.to(device)
    writer = SummaryWriter()
    
    num = 1
    for epoch in tqdm(range(1, args.epoch+1), desc='Training', unit='epoch'):
        losses = []
        for X, y in tqdm(train_loader, desc='Epoch {}'.format(epoch), unit='batch'):
            optimizer.zero_grad()
            if not args.conditioned:
                y = None
            else:
                y = y.to(device)
            if timeType == TimeType.ContinuousTimeLoss:
                loss = bfn.process_infinity(X.to(device), y)
            elif timeType == TimeType.DiscreteTimeLoss:
                loss = bfn.process_discrete(X.to(device), max_step=args.max_step)
            else:
                raise ValueError("The TimeType must be either ContinuousTimeLoss or DiscreteTimeLoss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
            
            optimizer.step()
            ema.update()
            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), num)
            num += 1
        scheduler.step()
        
        ave = 0
        for loss in losses:
            ave += loss
        ave = ave / len(losses)
        tqdm.write('Epoch {}: Loss: {:.8f}'.format(epoch, ave))
        writer.add_scalar('Loss/epoch_train', ave, epoch)
        if epoch % args.save_every_n_epoch == 0:
            torch.save(unet.state_dict(), f'./models/model_{epoch}.pth')
            tqdm.write('Epoch {}: saved: {}'.format(epoch, f'./models/model_{epoch}.pth'))

    torch.save(unet.state_dict(), args.save_model_path)
    tqdm.write('Epoch {}: saved: {}'.format(epoch, args.save_model_path))

def setup_train_common_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default="./models/model.pth")
    parser.add_argument("--save_every_n_epoch", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="./animeface")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Discrete Time Loss Option
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--conditioned", type=bool, default=False)
    return parser

