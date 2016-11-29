import config
import os, yaml
from IPython.core.debugger import Tracer

def set_configs(feature_configs=None, experiment_configs=None):
    configs = config.full_configs()
    configs.dump_configs()

def compute_features(config):
    import FeatureExtractor
    import 
    feature_extractor = FeatureExtractor.FeatureExtractor('resnet')
    output_path = config['path_to_video_features']
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    Tracer()()
    output = x.extract_features()

def load_configs(experiment_folder=None):
    '''
    load configs for default settings
    '''
    if not experiment_folder:
        configs = config.full_configs()
        configs.dump_configs()
        experiment_folder = configs.full_experiment_name

    return yaml.load(open(os.path.join(experiment_folder, 'config.yml')))

config = load_configs(experiment_folder=None)
compute_features(config)
