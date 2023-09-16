import random
from typing import Optional
from PIL import Image
import torchvision.transforms as transforms
import torch


def resize(h=32, w=32):
    return lambda img: img.resize((h, w), Image.BICUBIC)

def convert_to_3_channels(img):
    return img.convert('RGB')

def default_transform(h=32, w=32):
    return transforms.Compose([
        resize(h=h, w=w),
        convert_to_3_channels,
        transforms.ToTensor(),  # Convert images to tensors
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the pixel values to [-1, 1]
    ])
    
    
def set_seed(seed: Optional[int]):
    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)