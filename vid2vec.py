import os, sys, argparse, torch, cv2
import numpy as np
# from collections import namedtuple
from types import SimpleNamespace

from models.r21d.extract_r21d import ExtractR21D
from models.i3d.extract_i3d import ExtractI3D
from models.vggish.extract_vggish import ExtractVGGish
from utils.utils import sanity_check

MODEL_ZOO = [
    # r21d: after read_video, we keep running out of memory during transform()
    # {'name': 'r21d', 'feature_type': 'r21d', 'extractor': ExtractR21D},
    {'name': 'i3d', 'feature_type': 'i3d', 'extractor': ExtractI3D, 'ft_file_ext': '_rgb.npy'},
    {'name': 'vggish', 'feature_type': 'vggish', 'extractor': ExtractVGGish, 'ft_file_ext': '_vggish.npy'},
]

def VidInfo(vid_path):
	'''
	returns a dictonary of 'duration', 'fps', 'frame_count', 'frame_height', 'frame_width',
							'format', 'fourcc'
	'''
	vcap = cv2.VideoCapture(vid_path)
	info_dict = {
		'fps' : round(vcap.get(cv2.CAP_PROP_FPS),2), #int(vcap.get(cv2.CAP_PROP_FPS)),
		'frame_count': int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)), # number of frames should integars
		'duration': round(
			int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) / vcap.get(cv2.CAP_PROP_FPS),
			2), # round number of seconds to 2 decimals
		'frame_height': vcap.get(cv2.CAP_PROP_FRAME_HEIGHT),
		'frame_width': vcap.get(cv2.CAP_PROP_FRAME_WIDTH),
		'format': vcap.get(cv2.CAP_PROP_FORMAT),
		'fourcc': vcap.get(cv2.CAP_PROP_FOURCC)
	}
	vcap.release()
	return info_dict

def dict_to_obj(in_dict, object_name=None):
    # return namedtuple(object_name, in_dict.keys())(*in_dict.values())
    return SimpleNamespace(**in_dict)

def vid2vec(vid_path, model_name='id3',
    step_size = None, stack_size = None, extraction_fps = None,
    flow_type = 'pwc', one_stride = True,
    output_dir = './output/', debug = True):
    '''
    '''
    valid_models = [m['name'] for m in MODEL_ZOO]
    assert model_name in valid_models, f'{model_name} is not a valid model: {valid_models}'
    assert os.path.isfile(vid_path), f'{vid_path} is not found'
    assert torch.cuda.is_available(), f'GPU is required (for PWCNet)'
    assert flow_type in ['pwc','raft']

    vid_meta = VidInfo(vid_path)
    step_size = vid_meta['frame_count']-1 if one_stride else step_size
    stack_size = step_size if one_stride else stack_size
    if one_stride:
        print(f'''
        --- one-stride selected (make sure you have enough CUDA memory)
        \t using stack_size: {stack_size}
        \t using step_size: {step_size}
        ''')

    model_dict = [m for m in MODEL_ZOO if m['name']== model_name][0]
    args_dict = {
        'feature_type': model_dict['feature_type'],
        'streams': None, # uses PWCNet by default
        'video_paths': [vid_path],
        'file_with_video_paths': None,
        'flow_type': flow_type,         # pwc or raft
        'extraction_fps': extraction_fps,
        'step_size': step_size,
        'stack_size': stack_size,
        'show_pred': debug,
        'keep_tmp_files': False,
        'on_extraction': 'print' if debug else 'save_numpy',   #print, save_numpy, or save_pickle
        'output_path': output_dir,        # only for save_numpy or save_pickle
        'tmp_path': '/tmp/video_features/',           # only for custom fps
        'device_ids': [0]
    }
    extractor = model_dict['extractor']
    args = dict_to_obj(args_dict)#, object_name ='args')
    sanity_check(args)
    net = extractor(args)
    result = net(
        indices = torch.arange(1).to(device = torch.device('cuda:0'))
        )
    net.progress.close()

    if not debug:
        output_dir = os.path.join(output_dir, model_dict['feature_type'])
        vid_name, vid_ext = os.path.splitext(os.path.basename(vid_path))

        ft_fname = os.path.join(output_dir, vid_name + model_dict['ft_file_ext'] )
        assert os.path.isfile(ft_fname), f'feature extraction file missing: {ft_fname}'
        ft = np.load(ft_fname)
        print(f'features extracted ({ft.shape}): {ft_fname}')

        if model_name=='i3d':
            flow_fname = os.path.join(output_dir, vid_name + '_flow.npy')
            assert os.path.isfile(flow_fname), f'flow extraction missing: {flow_fname}'
            flow = np.load(flow_fname)
            print(f'flow features extracted ({flow.shape}): {flow_fname}')
            return ft, flow
        return ft
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description= f'vid2vec')
    parser.add_argument('--vid_path', required = True, type = str,
                        help = 'path to your video file')
    parser.add_argument('--model', required = False, type = str, default = 'i3d',
                        help = 'model_name [default: i3d]')
    parser.add_argument('--flow_type', required = False, type = str, default = 'pwc',
                        help = 'optical flow type; for i3d [default: pwc]')
    parser.add_argument('--step_size', required = False, type = int, default = None,
                        help = 'step size; for i3d [default: None]')
    parser.add_argument('--stack_size', required = False, type = int, default = None,
                        help = 'how many frames to pass into 3D Conv; for i3d [default: None]')
    parser.add_argument('--extraction_fps', required = False, type = int, default = None,
                        help = 'extraction FPS; for i3d [default: None]')
    parser.add_argument('--output_dir', required = False, type = str, default = './output/',
                        help = 'where to save extracted features [default: ./output/]')
    parser.add_argument('--one_stride', required = False, action = 'store_true', default = False,
                            help = 'pass whole video into 3D Conv; for i3d [default: False]')
    parser.add_argument('--debug', required = False, action = 'store_true', default = False,
                            help = 'show prediction and skip output files [default: False]')
    args = parser.parse_args()

    vid2vec(vid_path = args.vid_path, model_name = args.model,
        step_size= args.step_size,stack_size= args.stack_size,
        one_stride= args.one_stride, extraction_fps = args.extraction_fps,
        flow_type= args.flow_type,
        output_dir=args.output_dir, debug = args.debug)
