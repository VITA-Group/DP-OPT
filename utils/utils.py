import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)

