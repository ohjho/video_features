import os, sys, argparse, torch
import numpy as np

from models.r21d.extract_r21d import ExtractR21D
from modls.i3d.extract_i3d import ExtractI3D

MODEL_ZOO = [
    {'name': 'rgb', 'feature_type': 'r21d', 'extractor': ExtractR21D},
    {'name': 'i3d', 'feature_type': 'i3d', 'extractor': }
]

def dict_to_obj(in_dict, object_name):
    return namedtuple(object_name, in_dict.keys())(*in_dict.values())

def vid2vec(vid_path, model_name='id3'):
    '''
    '''
    assert model_name in [m['name'] for m in MODEL_ZOO], f'{model_name} is not a valid model'
    assert os.path.isfile(vid_path), f'{vid_path} is not found'

    model_dict = [m for m in MODEL_ZOO if o['name']== model_name][0]
    args_dict = {
        'feature_type': model_dict['feature_type'],
        'streams': None, # uses PWCNet by default
        'file_with_video_paths': [vid_path]
    }
    extractor = model_dict['extractor']
    args = dict_to_obj(args_dict, object_name ='args')
    result = extractor(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description= f'vid2vec')
    parser.add_argument('--vid_path', required = True, type = str,
                        help = 'path to your video file')
    parser.add_argument('--model', required = False, type = str, default = 'i3d',
                        help = 'model_name [default: i3d]')
    args = parser.parse_args()

    vid2vec(vid_path = args.vid_path, model_name = args.model)
