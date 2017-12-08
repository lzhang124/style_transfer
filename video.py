import neural_style
import scipy.misc
import numpy as np
from scipy import ndimage
from PIL import Image
# import imageio
import os
import math
from argparse import ArgumentParser

CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'vgg19.mat'
POOLING = 'max'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--map-colors',
            dest='map_colors', help='algorithm for mapping content colors to style image',
            metavar='MAP_COLORS')
    parser.add_argument('--luminance-transfer', action='store_true',
            dest='luminance_transfer', help='luminance transfer after style transfer')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    return parser

def extractFrames(inGif, outFolder):
    frame = Image.open(inGif)
    nframes = 0
    frames = []
    while frame:
        frame.save(outFolder+'/'+str(nframes)+'.png', 'PNG')
        frames.append(frame)
        nframes += 1
        try:
            frame.seek(nframes)
        except EOFError:
            break;
    return frames

def get_filenames(folder):
    filenames = [filename for filename in os.listdir(folder) if filename.endswith(".png")]
    def getint(name):
        basename = name.split('.')[0]
        return int(basename)
    filenames.sort(key=getint)
    return filenames

# def create_gif(name, inFolder):
#     filenames = get_filenames(inFolder)
#     images = []
#     for filename in filenames:
#         if filename.endswith(".png"): 
#             im = imageio.imread(inFolder+filename, 'PNG')
#             images.append(im)
#             continue
#         else:
#             continue
#     f = os.path.join(inFolder, name)
#     imageio.mimsave(f, images)

if __name__ == '__main__':
    # frames = extractFrames("content/panda.gif", "content/panda/tmp")
    prefix = "content/panda/tmp/"
    frames = get_filenames(prefix)
    parser = build_parser()
    options = parser.parse_args()

    options.initial = None
    options.content = prefix+frames[0]
    options.output = "content/panda/tmp/fp_0.jpg"
    for i in range(1,len(frames)):
        neural_style.main(parser, options)
        options.initial = prefix+frames[i-1]
        options.content = prefix+frames[i]
        options.output = "content/panda/tmp/fp_"+str(i)+".jpg"


