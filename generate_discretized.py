import argparse
import torch
from generate_common import common_generate, setup_generate_common_parser
from train_common import BFNType, TimeType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bfnType = BFNType.Discretized
timeType = TimeType.DiscreteTimeLoss

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = setup_generate_common_parser(parser)
    return parser
    
if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    common_generate(args, bfnType = bfnType, timeType=timeType)