#!/usr/bin/python
# End-to-End Memory Networks -- Theano implementation
# Modified to run with on the MovieQA dataset

# General imports
import os
import sys
import ipdb
import json
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import time
import word2vec as w2v
from collections import Counter
from optparse import OptionParser
# Local imports
from movieqa_importer import MovieQA
from word_featurizer import sent2q_featurizer
#from memN2N_text import MemoryNetwork
import utils

# Seed random number generators
rng = np.random
rng.seed(1234)

w2v_mqa_model_filename = 'models/movie_plots_1364.d-300.mc1.w2v'


def get_minibatch(batch_idx, stM, quesM, ansM, qinfo, mute_targets=False):
    """Create one mini-batch from the data.
    Inputs:
        batch_idx - a vector of indices to select stories, questions from

    Returns:
        memorydata - batchsize X numsentence X numwords - story representation
        inputq - batchsize X numwords - input question representation
        target - batchsize X 1 - correct answer indices during training 0-4
        multians - batchsize X 5 X numwords - multiple choice answer representations
        b_qinfo - batchsize X [] - list of question information
    """

    story_shape = stM.values()[0].shape
    num_ma_opts = ansM.shape[1]

    inputq = np.zeros((len(batch_idx), quesM.shape[1]), dtype='int32')                             # question input vector
    target = np.zeros((len(batch_idx)), dtype='int32')                                             # answer (as a single number)
    memorydata = np.zeros((len(batch_idx), story_shape[0], story_shape[1]), dtype='int32')         # memory statements
    multians = np.zeros((len(batch_idx), num_ma_opts, ansM.shape[2]), dtype='int32')               # multiple choice answers
    b_qinfo = []

    for b, bi in enumerate(batch_idx):
        # question vector
        inputq[b] = quesM[bi]
        # answer option number
        if not mute_targets:
            target[b] = qinfo[bi]['correct_option']
        # multiple choice answers
        multians[b] = ansM[bi]   # get list of answers for this batch
        # get story data
        memorydata[b] = stM[qinfo[bi]['movie']]
        # qinfo
        b_qinfo.append(qinfo[bi])

    
    return memorydata, inputq, target, multians, b_qinfo


def count_errors(yhat, target):
    """Counts the total number of errors.
    """

    errors = sum(np.argmax(yhat, axis=1) != target)
    return errors


def call_train_epoch(train_func, train_data, train_range, bs=8, lr=0.01, iterprint=False):
    """One epoch of training.
    """

    train_error, train_cost, it = 0., 0., 0

    n_train_batches = int(len(train_range) / bs)
    train_perm = rng.permutation(train_range)
    # iterate over all batches in the data
    for batch_count in xrange(n_train_batches):
        it += 1
        # get indices of this minibatch
        this_batch = train_perm[batch_count * bs : (batch_count + 1) * bs]
        # get minibatch
        memorydata, inputq, target, multians, b_qinfo = \
            get_minibatch(this_batch, train_data['s'], train_data['q'], train_data['a'], train_data['qinfo'])
        # call train model
        cost, yhat, g_norm, p_norm = train_func(memorydata, inputq, target, multians, lr)
        er = count_errors(yhat, target)

        # print iteration info
        if iterprint:
            print "\titer: %5d | train error: %7.3f | batch-cost: %7.3f" %(it, 100.0*er/bs, cost),
            print "| W norms:", p_norm, "| G norms:", g_norm

        # accumulate stuff
        train_cost += cost
        train_error += er
        # train_gnorm += g_norm

    # normalize counts over all batches and samples
    train_error = 100 * train_error / (n_train_batches * bs)
    train_cost = train_cost / n_train_batches

    return train_error, train_cost, it


def call_test(test_func, test_data, data_range=None, bs=8):
    """Run one round of test on all data.
        if data_range is None:
            we are running cleanly on the actual val, or test sets
        else:
            we are running on train-val
    """

    if data_range:  # train-val
        num_qa = len(data_range)
        n_test_batches = int(len(data_range) / bs)
        mute_targets = False
        test_error = 0.
    else:  # val and test
        num_qa = len(test_data['qinfo'])
        n_test_batches = int(np.ceil(1.0*num_qa / bs))
        mute_targets = True
        ans_keys = {}

    # iterate over all batches in the data
    for batch_count in xrange(n_test_batches):
        # get indices of this minibatch
        if data_range:  # train-val
            this_batch = data_range[batch_count * bs : (batch_count + 1) * bs]
        else:  # val and test
            this_batch = range(batch_count * bs, min( (batch_count+1) * bs, len(test_data['qinfo']) ))

        # get minibatch
        memorydata, inputq, target, multians, b_qinfo = \
            get_minibatch(this_batch, test_data['s'], test_data['q'], test_data['a'], test_data['qinfo'], mute_targets=mute_targets)
        # call test function
        yhat = test_func(memorydata, inputq, multians)
        if data_range:  # train-val
            er = count_errors(yhat, target)
            test_error += er
        else:  # val and test
            ans_keys.update({qa['qid']: np.argmax(yhat[k]) for k, qa in enumerate(b_qinfo)})

    if data_range:  # train-val
        test_error = 100 * test_error / (n_test_batches * bs)
        return test_error
    else:  # val and test
        return ans_keys


def create_vocabulary(QAs, stories, v2i, w2v_vocab=None, word_thresh=2):
    """Create the vocabulary by taking all words in stories, questions, and answers taken together.
    Also, keep only words that appear in the word2vec model vocabulary (if provided with one).
    """

    print "Creating vocabulary.",
    if w2v_vocab is not None:
        print "Adding words based on word2vec"
    else:
        print "Adding all words"
    # Get all story words
    all_words = [word for story in stories for sent in story for word in sent]

    # Parse QAs to get actual words
    QA_words = []
    for QA in QAs:
        QA_words.append({})
        QA_words[-1]['q_w'] = utils.normalize_alphanumeric(QA.question.lower()).split(' ')
        QA_words[-1]['a_w'] = [utils.normalize_alphanumeric(answer.lower()).split(' ') for answer in QA.answers]

    # Append question and answer words to all_words
    for QAw in QA_words:
        all_words.extend(QAw['q_w'])
        for answer in QAw['a_w']:
            all_words.extend(answer)

    # threshold vocabulary, at least N instances of every word
    vocab = Counter(all_words)
    vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

    # create vocabulary index
    for w in vocab:
        if w not in v2i.keys():
            if w2v_vocab is None:
                # if word2vec is not provided, just dump the word to vocab
                v2i[w] = len(v2i)
            elif w2v_vocab is not None and w in w2v_vocab:
                # check if word in vocab, or else ignore
                v2i[w] = len(v2i)

    print "Created a vocabulary of %d words. Threshold removed %.2f %% words" \
            %(len(v2i), 100*(1. * len(set(all_words)) - len(v2i))/len(all_words))

    return QA_words, v2i


def data_in_matrix_form(stories, QA_words, v2i):
    """Make the QA data set compatible for memory networks by
    converting to matrix format (index into LUT vocabulary).
    """

    def add_word_or_UNK():
        if v2i.has_key(word):
            return v2i[word]
        else:
            return v2i['UNK']

    # Encode stories
    max_sentences = max([len(story) for story in stories.values()])
    max_words = max([len(sent) for story in stories.values() for sent in story])

    storyM = {}
    for imdb_key, story in stories.iteritems():
        storyM[imdb_key] = np.zeros((max_sentences, max_words), dtype='int32')
        for jj, sentence in enumerate(story):
            for kk, word in enumerate(sentence):
                storyM[imdb_key][jj, kk] = add_word_or_UNK()

    print "#stories:", len(storyM)
    print "storyM shape (movie 1):", storyM.values()[0].shape

    # Encode questions
    max_words = max([len(qa['q_w']) for qa in QA_words])
    questionM = np.zeros((len(QA_words), max_words), dtype='int32')
    for ii, qa in enumerate(QA_words):
        for jj, word in enumerate(qa['q_w']):
            questionM[ii, jj] = add_word_or_UNK()
    print "questionM:", questionM.shape

    # Encode answers
    max_answers = max([len(qa['a_w']) for qa in QA_words])
    max_words = max([len(a) for qa in QA_words for a in qa['a_w']])
    answerM = np.zeros((len(QA_words), max_answers, max_words), dtype='int32')
    for ii, qa in enumerate(QA_words):
        for jj, answer in enumerate(qa['a_w']):
            if answer == ['']:  # if answer is empty, add an 'UNK', since every answer option should have at least one valid word
                answerM[ii, jj, 0] = 1
                continue
            for kk, word in enumerate(answer):
                answerM[ii, jj, kk] = add_word_or_UNK()
    print "answerM:", answerM.shape

    return storyM, questionM, answerM


def associate_additional_QA_info(QAs):
    """Get some information about the questions like story index and correct option.
    """

    qinfo = []
    for QA in QAs:
        qinfo.append({'qid':QA.qid,
                      'movie':QA.imdb_key,
                      'correct_option':QA.correct_index})
    return qinfo


def normalize_documents(stories, normalize_for=('lower', 'alphanumeric'), max_words=40):
    """Normalize all stories in the dictionary, get list of words per sentence.
    """

    for movie in stories.keys():
        for s, sentence in enumerate(stories[movie]):
            sentence = sentence.lower()
            if 'alphanumeric' in normalize_for:
                sentence = utils.normalize_alphanumeric(sentence)
            sentence = sentence.split(' ')[:max_words]
            stories[movie][s] = sentence
    return stories

def seq2seq_preprocessor(story, question, qinfo):
    '''
    For every question, map each sentence in the story
    to it.
    '''
    n_questions = question.shape[0]
    max_question_words = question.shape[1]
    max_story_words = story.values()[0].shape[1]
    question_input = np.zeros((0, max_question_words), dtype=np.int32)
    story_input = np.zeros((0, max_story_words))
    for question_id in xrange(n_questions):
        current_question = question[question_id]
        current_info = qinfo[question_id]
        current_story = story[current_info['movie']]
        n_sentences_indicator = np.sum(current_story, axis=1).tolist()
        n_sentences = sum([x>0 for x in n_sentences_indicator])
        current_story = current_story[:n_sentences]
        current_question = np.repeat(current_question[np.newaxis], n_sentences, axis=0)
        question_input = np.concatenate((question_input, current_question))
        story_input = np.concatenate((story_input, current_story))
        if (question_id+1) % 100 == 1:
            print 'Generating inputs for %dth question'%(question_id)
    return story_input, question_input

def main(options):
    """Main function which wraps everything.
        - Prepare data: word2vec, vocabulary creation, train/val/test splits
        - Build MemoryNetwork Theano model
        - Run training pass
    """

    print "----------- Prepare data ------------"
    # Get list of MAs and movies
    mqa = MovieQA.DataLoader()

    ### Process story source
    stories, QAs = mqa.get_story_qa_data('full', options['data']['source'])
    stories = normalize_documents(stories)

    ### Load Word2Vec model
    w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')
    options['memnn']['w2v'] = w2v_model
    options['memnn']['d-w2v'] = len(w2v_model.get_vector(w2v_model.vocab[1]))
    print "Loaded word2vec model: dim = %d | vocab-size = %d" \
        %(options['memnn']['d-w2v'], len(w2v_model.vocab))

    ### Create vocabulary-to-index and index-to-vocabulary
    v2i = {'': 0, 'UNK':1}  # vocabulary to index
    QA_words, v2i = create_vocabulary(QAs, stories, v2i,
                                 w2v_vocab=w2v_model.vocab.tolist(),
                                 word_thresh=options['data']['vocab_threshold'])
    i2v = {v:k for k,v in v2i.iteritems()}

    ### Convert QAs and stories into numpy matrices (like in the bAbI data set)
    # storyM - Dictionary - indexed by imdb_key. Values are [num-sentence X max-num-words]
    # questionM - NP array - [num-question X max-num-words]
    # answerM - NP array - [num-question X num-answer-options X max-num-words]
    storyM, questionM, answerM = data_in_matrix_form(stories, QA_words, v2i)
    qinfo = associate_additional_QA_info(QAs)

    ### Build preprocessor
    preprocessor_story, preprocessor_question = seq2seq_preprocessor(storyM, questionM, qinfo)
    preprocessor_story = preprocessor_story.astype('int32')
    preprocessor_question = preprocessor_question.astype('int32')
    input_dim = len(w2v_model.vocab)
    maxlen = preprocessor_story.shape[1]
    maxqlen = preprocessor_question.shape[1]
    model_filename = 'model_' + options['data']['source'] + '.h5'
    embed_dim = 256
    preprocessor = sent2q_featurizer(input_dim, embed_dim, model_filename, maxlen=maxlen, maxqlen=maxqlen)
    preprocessor.train(preprocessor_story, preprocessor_question)
    del preprocessor
    preprocessor = sent2q_featurizer(input_dim, embed_dim, model_filename, test_time=True)
    story_features = {}
    for key_id, keyname in enumerate(storyM.keys()):
        current_story = storyM[keyname]
        story_features[keyname] = preprocessor.compute_features(current_story)
        if key_id % 10 == 0:
            print 'Extracting features for %dth movie' % key_id
    import pickle as pkl
    pkl_filename = 'sent2q_' + options['data']['source'] + '.pkl'
    with open(pkl_filename, 'w') as f:
        pkl.dump(story_features, f)
    f.close()

def init_option_parser():
    """Initialize parser.
    """

    usage = """
    Important options are printed here. Check out the code more tweaks.
    %prog -s <story_source> [-z <evaluation_set>] [-n <num_mem_layers>]
                    [--learning_rate <lr>] [--batch_size <bs>] [--nepochs <ep>]
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--story_source", action="store", type="string", default="",
                      help="Story source text: split_plot | dvs | subtitle | script")
    parser.add_option("-z", "--evaluation_set", action="store", type="string", default="val",
                      help="Run final evaluation on? [val] | test")
    parser.add_option("-n", "--num_mem_layers", action="store", type=int, default=1,
                      help="Number of Memory layers")
    parser.add_option("",   "--batch_size", action="store", type=int, default=8,
                      help="Batch size. Ranges from 8 to 64 (depends on GPU memory)")
    parser.add_option("",   "--learning_rate", action="store", type=float, default=0.01,
                      help="Initial learning rate for SGD")
    parser.add_option("",   "--nepochs", action="store", type=int, default=100,
                      help="Train for N epochs")
    return parser


if __name__ == '__main__':
    ### Parse command line options
    parser = init_option_parser()
    opts, args = parser.parse_args(sys.argv)

    assert opts.story_source in ['split_plot', 'dvs', 'subtitle', 'script'], \
        utils.fail_nicely("Invalid story type", parser)
    print 'Evaluating Memory Networks (Text) on MovieQA using: %s' % opts.story_source

    # -------------------------------------------------------
    # Initialize options, lots of defaults, some from parser
    # -------------------------------------------------------
    options = {'memnn':{}, 'train':{}, 'data':{}}
    # MemN2N options
    options['train']['nepochs'] = opts.nepochs                  # number of train epochs
    options['train']['batch_size'] = opts.batch_size            # batch size
    options['train']['learning_rate'] = opts.learning_rate      # learning rate
    options['train']['gnorm'] = {'max_norm': 40}                # gradient normalization options 'max_norm' OR 'clip'
    # Data options

    ### Deprecated - keep code simpler!
    # options['mode'] = 'multi_choice'                          # QA mode, 'single' vs. 'multi_choice'
    # options['memnn']['position_encode'] = False               # encode position by weighting words in a sentence
    # options['memnn']['temporal_encode'] = False               # encode time by adding extra "word" at end of sentence
    # options['memnn']['randomize_time'] = 0.1                  # make the time encoding a bit noisy
    # options['memnn']['l2_regularize'] = False                 # L2 regularization on the parameters
    # options['memnn']['weight_sharing'] = "all"                # share weights between layers? types: 'adjacent', 'rnn', 'all'
    # options['memnn']['replace_LUT_with_w2v'] = True           # THIS IS THE CASE! replace LUT with word2vec initialized vectors, use extra embedding
    # options['memnn']['init_LUT'] = 'randn'                    # weight initialization method for LUT based embeddings: 'randn', 'w2v'

    # -------------------------------------------------------

    # Go, go, go!
    main(options)
    sys.exit()
