import keras
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions
from models import *
from keras.models import Model
from IPython.core.debugger import Tracer

class FeatureExtractor:
    def __init__(self, featurename, args=None):
        if not getattr(self, 'init_' + featurename):
            raise(NotImplemented)
        else:
            getattr(self, 'init_' + featurename)()

    def init_resnet(self):
        from models.resnet50 import ResNet50
        self.base_model = ResNet50(weights='imagenet', include_top=False)
        self.input_shape = 224
        self.feature_dim = 2048

    def init_vggnet(self):
        from models.vgg16 import VGG16
        self.base_model = VGG16(weights='imagenet', include_top=False)
        self.input_shape = 224
        self.feature_dim = 4096

    def init_inception(self):
        from models.inception_v3 import InceptionV3
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        self.input_shape = 299
        self.feature_dim = 2048
    
    def preprocess(self, input_img,input_type):
        if input_type == 'Image':
            return image.load_img(input_img)[..., np.newaxis]
        else:
            video_file = cv2.VideoCapture(input_img)
            frames = []
            while(True):
                frame = video_file.read()
                if frame[0]:
                    frames.append(frame[1][..., np.newaxis])
                else:
                    return np.concatenate(frames, axis=3)

    def check_input(self, input_img):
        if isinstance(input_img, str):
            if cv2.imread(input_img).shape:
                return 'Image'
            else:
                video_file = cv2.VideoCapture(input_img)
                input_img = [video_file.read()[1]]
                if not input_img[0].shape:
                    raise(ValueError('Could not read input as image or video.'))
                return 'video'
            input_img = [input_img]

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
            target_img[i] = preprocess_input(target_image[i])

        return target_img

    def extract_features(self, input_img):
        if isinstance(input_img, str):
            img_type = self.check_input(input_img)
            input_img = self.preprocess(input_img, img_type)
        elif isinstance(input_img, list):
            input_img = self.check_input(input_img)
        if input_img.shape[0] != self.input_shape:
            input_img = self.resize_to_input_shape(input_img, self.input_shape)

        return self.model.predict(input_img)


'''
if __name__ == '__main__':
    x = FeatureExtractor('resnet')
    x.extract_features()
'''
