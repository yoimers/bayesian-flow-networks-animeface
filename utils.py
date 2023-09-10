from PIL import Image
import torchvision.transforms as transforms


def resize(h=32, w=32):
    return lambda img: img.resize((h, w), Image.BICUBIC)

def convert_to_3_channels(img):
    return img.convert('RGB')

