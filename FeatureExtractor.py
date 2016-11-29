import keras
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions
from models import *
from IPython.core.debugger import Tracer

class FeatureExtractor:
    def __init__(self, featurename, args=None):
        if not getattr(self, 'init_' + featurename):
            raise(NotImplemented)
        else:
            getattr(self, 'init_' + featurename)()

    def init_resnet(self):
        from models.resnet50 import ResNet50
        self.model = ResNet50(weights='imagenet')

    def init_vggnet(self):
        from models.vgg16 import VGG16
        self.model = VGG16(weights='imagenet')

    def init_inception(self):
        from models.inception_v3 import InceptionV3
        self.model = InceptionV3(weights='imagenet')

    def extract_features(self):
        self.model.include_top = False

'''
if __name__ == '__main__':
    x = FeatureExtractor('resnet')
    x.extract_features()
'''
