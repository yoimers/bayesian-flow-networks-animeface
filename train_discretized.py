import argparse
from train_common import BFNType, TimeType, setup_train_common_parser, train

bfnType=BFNType.Discretized
timeType=TimeType.ContinuousTimeLoss

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = setup_train_common_parser(parser)
    return parser
    
if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    train(args, bfnType=bfnType, timeType=timeType)