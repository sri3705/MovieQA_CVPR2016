import keras
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions
from models import *
from keras.models import Model
from IPython.core.debugger import Tracer
import os, cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, featurename, args=None):
        if not getattr(self, 'init_' + featurename):
            raise(NotImplemented)
        else:
            getattr(self, 'init_' + featurename)()
        self.output_shape = list(self.output_shape)
        self.output_shape[0] = 0
        assert(self.output_shape[-1])
        self.output_shape = [1 if x is None else x for x in self.output_shape]
        self.output_shape = tuple(self.output_shape)

    def init_resnet(self):
        from models.resnet50 import ResNet50
        self.base_model = ResNet50(weights='imagenet', include_top=False)
        self.input_shape = 224
        self.feature_dim = 2048
        self.output_shape = self.base_model.output_shape

    def init_vggnet(self):
        from models.vgg16 import VGG16
        self.base_model = VGG16(weights='imagenet', include_top=False)
        self.input_shape = 224
        self.feature_dim = 4096
        self.output_shape = self.base_model.output_shape

    def init_inception(self):
        from models.inception_v3 import InceptionV3
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        self.input_shape = 299
        self.feature_dim = 2048
        self.output_shape = self.base_model.output_shape

    def preprocess(self, input_img, input_type):
        if input_type == 'Image':
            input_img = image.load_img(input_img)[..., np.newaxis]
            if input_img.shape[0] != self.input_shape:
                input_img = self.resize_to_input_shape(input_img, self.input_shape)
            return self.base_model.predict(input_img)
        else:
            video_file = cv2.VideoCapture(input_img)
            output = np.zeros(self.output_shape)
            max_frames_to_process = 1000
            current_input = np.zeros((self.input_shape, self.input_shape, 3, 0))
            frames = []
            while(True):
                frame = video_file.read()
                if frame[0]:
                    if len(frames) < max_frames_to_process:
                        frames.append(frame[1][..., np.newaxis])
                    else:
                        current_input = self.resize_to_input_shape(np.concatenate(frames, axis=3), self.input_shape)
                        current_output = self.base_model.predict(current_input)
                        output = np.concatenate((output, current_output))
                        current_input = np.zeros((self.input_shape, self.input_shape, 3, 0))
                        frames = []
                else:
                    if not frames: return output
                    current_input = self.resize_to_input_shape(np.concatenate(frames, axis=3), self.input_shape)
                    current_output = self.base_model.predict(current_input)
                    return np.concatenate((output, current_output))

    def check_input(self, input_img):
        if isinstance(input_img, unicode):
            try:
                cv2.imread(input_img).shape
                return 'Image'
            except:
                video_file = cv2.VideoCapture(input_img)
                test_frame = video_file.read()[1]
                try:
                    test_frame.shape
                    return 'video'
                except:
                    print 'Could not read input as image or video: %s'%input_img
                    failed_cases = open('feature_extractor_failures', 'a')
                    failed_cases.write(input_img + '\n')
                    failed_cases.close()
                    #raise(ValueError('Could not read input as image or video.'))
            #input_img = [input_img]

        elif isinstance(input_img, list):
            try:
                input_img.shape[0]
            except:
                raise(TypeError('expected list of images as input'))
            try:
                input_img = map(lambda x: x[..., np.newaxis], input_img)
                input_img = np.concatenate(input_img, axis=3)
                return input_img
            except:
                raise(ValueError('Expected list of images to be of same shape.'))
        elif type(input_img).__module__ == 'numpy':
            pass
        else:
            raise(TypeError('Expected input to be a string or a list of strings'))

    def resize_to_input_shape(self, input_img, shape=224):
        from PIL import Image
        n_frames = input_img.shape[-1]
        target_img = np.zeros((n_frames, shape, shape, 3))
        for i in xrange(n_frames):
            target_img[i] = image.img_to_array(Image.fromarray(input_img[..., i]).resize((shape, shape)))
        target_img = preprocess_input(target_img)

        return target_img

    def extract_features(self, inputs):
        from copy import deepcopy
        from IPython.core.debugger import Tracer
        input_img = deepcopy(inputs[0])
        print 'extracting features for %s...' % (os.path.basename(input_img))
        save_output_to = deepcopy(inputs[1])
        if os.path.isfile(save_output_to) or os.path.isfile(save_output_to + '.npy'):
            return
        if isinstance(input_img, unicode):
            img_type = self.check_input(input_img)
            if not img_type: return
        else:
            raise(NotImplemented)

        output = self.preprocess(input_img, img_type)
        if not save_output_to:
            return output
        else:
            output_root = os.path.dirname(save_output_to)
            if not os.path.isdir(output_root): os.makedirs(output_root)
            np.save(save_output_to, output)
            return

'''
if __name__ == '__main__':
    x = FeatureExtractor('resnet')
    x.extract_features()
'''
