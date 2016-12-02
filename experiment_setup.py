import config
import os, yaml
from IPython.core.debugger import Tracer
from multiprocessing import Pool
import progressbar as pb
from encode_qa_and_text import check_save_directory
import utils
def set_configs(feature_configs=None, experiment_configs=None):
    configs = config.full_configs()
    configs.dump_configs()

def compute_features(config, mp=False):
    import FeatureExtractor
    import movieqa_importer
    feature_extractor = FeatureExtractor.FeatureExtractor(config['video_features'])
    mqa = movieqa_importer.MovieQA.DataLoader()
    vl_qa, _ = mqa.get_video_list('full', 'all_clips')
    desc = config['video_features']
    document_type = 'video_clips'
    check_save_directory(filename=utils.DOC_DESC_TEMPLATE %(document_type, desc, ''))
    videos = []
    outputs = []
    for i, imdb_key in enumerate(vl_qa.keys()):
        current_videos = map(lambda x: os.path.join('/cs/vml4/smuralid/datasets/movieqa/MovieQA_benchmark','story', document_type, imdb_key, x), vl_qa[imdb_key])
        videos.extend(current_videos)
        output = map(lambda x: os.path.join((utils.DOC_DESC_TEMPLATE % (document_type, desc, imdb_key))[:-4], x.replace('.mp4', '')), vl_qa[imdb_key])
        outputs.extend(output)
    if mp:
        inputs = [[x, y] for x,y in zip(videos, outputs)]
        from random import shuffle
        shuffle(inputs)
        #pool = Pool(mp)
        #pool.map(feature_extractor.extract_features, inputs)
        for x in inputs: feature_extractor.extract_features(x)
    else:
        for video, output in zip(videos, outputs):
            feature_extractor.extract_features([video, output])



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
compute_features(config, mp=1)
