from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers import recurrent
import numpy as np
import h5py

'''
Training using cho et al. 2014 sequence to sequence model to improve feature representations of sentences of a story
Note: GRU needs to be used both on encoder and decoder sides
'''
class sent2q_featurizer:
    def __init__(self, input_dim, embed_dim, model_filename, maxqlen=25, maxlen=100, batch_size=128, hiddensize=256, n_encoderlayers=1, n_decoderlayers=1, celltype='GRU', maxepochs=10):
        self.input_dim = input_dim
        self.maxlen = maxlen#on the encoder side
        self.maxqlen = maxqlen
        self.batch_size = batch_size
        self.celltype = getattr(recurrent, celltype)
        self.input_dim = self.input_dim
        self.hiddensize = hiddensize
        self.n_encoderlayers = n_encoderlayers
        self.n_decoderlayers = n_decoderlayers
        self.embed_dim = embed_dim
        self.maxepochs=maxepochs
        self.model_filename = model_filename


    def build_model(self):
        print('Building model...')
        self.model = Sequential()
        self.model.add(Embedding(self.input_dim, self.embed_dim, mask_zero=True, input_length=self.maxlen))

        #adding encoders
        if self.n_encoderlayers < 2:
            self.model.add(self.celltype(self.hiddensize, return_sequences=False))
        else:
            self.model.add(self.celltype(self.hiddensize, return_sequences=False))
            for i in xrange(self.n_encoderlayers - 2):
                self.model.add(self.celltype(self.hiddensize, return_sequences=False))
            self.model.add(self.celltype(self.hiddensize, return_sequences=False))
        #repeating over sequence
        self.model.add(RepeatVector(self.maxqlen))

        #adding decoders
        if self.n_decoderlayers < 2:
            self.model.add(self.celltype(self.hiddensize, return_sequences=True))
        else:
            self.model.add(self.celltype(self.hiddensize, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.input_dim)))
        self.model.add(Activation('softmax'))

    def question_to_one_hot(self, question):
        max_words = question.shape[1]
        n_questions = question.shape[0]
        output = np.zeros((question.shape[0], max_words, self.input_dim))
        for question_id in xrange(n_questions):
            for word_id in xrange(max_words):
                idx = question[question_id, word_id]
                output[question_id, word_id, idx] = 1
        return output

    def train(self, story, question):
        self.build_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for epoch in xrange(self.maxepochs):
            print 'epoch ---------> ' + str(epoch)
            n_batches = np.floor(story.shape[0] / self.batch_size)
            for current_batch in xrange(int(n_batches)):
                start_idx = current_batch*self.batch_size
                end_idx = (current_batch+1)*self.batch_size
                current_story = story[start_idx:end_idx]
                current_question = self.question_to_one_hot(question[start_idx:end_idx])
                self.model.fit(current_story, current_question, batch_size=self.batch_size, nb_epoch=1)
            self.model.save(self.model_filename)

    def compute_features(self, story):
        from keras.models import load_model
        self.model = load_model(self.model_filename)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        base_model = Model(input=self.model.input, output=self.model.get_layer('repeatvector1').input)
        return base_model.predict(story)
