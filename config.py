import yaml
'''
The idea is to set training and testing options.
This file needs to have the following configs:
    1. Feature configs
        1.1 Path to saved/to-be-saved Features
        1.2 Feature Name
    2. Training configs
        All the training parameters
        2.1 Initialization params
        2.2 lr settings
    3. Model configs
        3.1 architecture
    4. General configs
        4.1 Path to Experiment File
        4.2 Experiment name inclusions
'''

class supersub_configs:
    def non_default_attributes(self, configs=None):
        if not configs:
            import inspect
            attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
            return [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        else:
            return configs.items()

        #remove __ ones

    def configs_to_string(self, ignore_attribs=None):
        '''
        attributes to string (non-methods, non-default ones)
        ignore_attribs: list of attributes to be ignored
        '''
        attributes = self.non_default_attributes(self.configs)
        if ignore_attribs: attributes = [a for a in attributes if not(a[0] in ignore_attribs)]
        self.attributes = dict(attributes)
        self.attributes_to_string = yaml.dump(self.attributes, default_flow_style=False)


    def save_configs(self, filename, ignore_attribs=None):
        self.configs_to_string(ignore_attribs=ignore_attribs)
        #pretty dumping rather than dict dumping 
        with open(filename, 'w') as outfile:
            outfile.write(self.attributes_to_string)
        outfile.close()

class feature_configs(supersub_configs):
    def __init__(self, text_features='wordnet', video_features='resnet', prefix_path_to_features=None):
        '''
        video features: resnet/inception/vgg
        text_features: wordnet
        path_to_feature: prefix path to features
        '''
        self.video_features = video_features
        self.text_features = text_features
        if not prefix_path_to_features:
            self.prefix_path_to_features = '/cs/vml4/smuralid/datasets/movieqa/features'
        else:
            self.prefix_path_to_features = prefix_path_to_features
        self.path_to_video_features = self.compute_path_to_features(self.video_features)
        self.path_to_text_features = self.compute_path_to_features(self.text_features)


    def compute_path_to_features(self, featurename):
        import os
        return os.path.join(self.prefix_path_to_features, featurename)

    def check_feature_path_exists(self, path_to_features):
        import os
        import warnings
        if not os.path.isdir(path_to_features):
            warnings.warn('Features do not exist in the specified path.')
        else:
            print('Loading features from %s'%path_to_features)


class experiment_configs(supersub_configs):
    def __init__(self, lr=0.001, batch_size=16, max_epochs=10, rnn_init='uniform', rnn_units='lstm', \
                text_rnn_size=1000, video_rnn_size=1000, n_stacks=2, decay_rate=1e-5, tol=1e-4, early_stop=2, prefix_path_to_experiment=None):
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.rnn_init = rnn_init
        self.rnn_units = rnn_units
        self.decay_rate = decay_rate
        self.tol = tol
        self.early_stop = early_stop
        self.video_rnn_size = video_rnn_size
        self.text_rnn_size = text_rnn_size
        if not prefix_path_to_experiment:
            self.prefix_path_to_experiment = '/local-scratch/srikanth/NMT-project'
        else:
            self.prefix_path_to_experiment = path_to_experiment



class full_configs(supersub_configs):

    def init_class(self, classname, input_args):
        import inspect
        obj = classname()
        classname_str = classname.__name__
        class_attribs = obj.non_default_attributes()
        for key in input_args.keys():
            if key in class_attribs:
                setattr(obj, key, class_args[key])
            else:
                raise(ValueError('No such configuration option found for %s: %s'%(classname_str, key)))
        return obj

    def __init__(self, feature_config={}, experiment_config={}):
        '''
        Note:both feature_configs, and experiment_configs should be dicts
        '''
        self.feature_configs = self.init_class(feature_configs, feature_config)
        self.experiment_configs = self.init_class(experiment_configs, experiment_config)
        self.configs = dict(self.feature_configs.non_default_attributes() + self.experiment_configs.non_default_attributes())


    def set_experiment_name(self, include_attribs=None):
        import os
        if not include_attribs: include_attribs=['lr', 'batch_size', 'video_features', 'text_features',
                            'rnn_units', 'video_rnn_size', 'text_rnn_size']
        ignore_attribs = list(set(self.configs.keys()) - set(include_attribs))
        self.configs_to_string(ignore_attribs=ignore_attribs)
        self.full_experiment_name = os.path.join(self.configs['prefix_path_to_experiment'], self.attributes_to_string.replace('\n', ','))
        self.full_experiment_name = ''.join(self.full_experiment_name.replace(':', '=').split())[:-1]


    def dump_configs(self, include_attribs_exp_path=None):
        import os
        self.set_experiment_name(include_attribs_exp_path)
        if not os.path.isdir(self.full_experiment_name):
            os.makedirs(self.full_experiment_name)
        out_yaml = os.path.join(self.full_experiment_name, 'config.yml')
        self.save_configs(out_yaml)
