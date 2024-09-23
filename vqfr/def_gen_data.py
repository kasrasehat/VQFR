from os import path as osp
import sys
sys.path.append("/home/user1/kasra/pycharm-projects/VQFR")
from vqfr.data.LQImageGenerator_def import LowQualityImageGeneratorV2
import argparse
import yaml
from collections import OrderedDict

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


parser = argparse.ArgumentParser()
parser.add_argument('-opt', default='/home/user1/kasra/pycharm-projects/VQFR/options/train/VQFR/generate_low_quality_images.yml', type=str, required=False, help='Path to option YAML file.')
args = parser.parse_args()

# parse yml to dict
with open(args.opt, mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])
    
   
generator = LowQualityImageGeneratorV2(opt['datasets'])
generator.generate_lq_images()
    
    