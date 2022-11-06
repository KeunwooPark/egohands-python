import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--egohands_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()

def main(args):
    egohands_path = Path(args.egohands_dir)
    
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)