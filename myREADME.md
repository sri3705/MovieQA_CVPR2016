To replicate the different models and results please follow the instructions below:

1. Register and download MovieQA dataset - Details in http://movieqa.cs.toronto.edu/register/. Follow instructions there
in order to setup the dataset paths and folders appropriately.
2. Add the dataset folder to path
Change the path in <code>movieqa_importer.py</code>.
3. Compute the features using sent2q_preprocess.py. You could change inputs: split_plot / script using '-s' option. Trains the
sequence to sequence model of sentences-question for 10 epochs, and computes the features, saving it as './sent2q_split_plot.pkl', or './sent2q_script.pkl' 
4. Finally, run the code, memory_network_model.py with the appropriate -s option feeded in the previopus step. It also has other options, including
memory network parameters, and word2vec input file. P

for more details please refer to:

<code>python memory_network_model.py -h</code>

----

Baseline models

To replicate the baseline model, just use memory_network_text.py. It has almost all options as memory_network_text.

for more details please refer to:

<code>python memory_network_text.py -h</code>

##### Prerequisities
+ Word2Vec model trained on 1364 movie plot synopses. [Download here](https://cvhci.anthropomatik.kit.edu/~mtapaswi/downloads/movie_plots_1364.d-300.mc1.w2v) and store to "models" folder
+ Skip-Thought encoder. [Github repo](https://github.com/ryankiros/skip-thoughts)
Please follow instructions on that repository.
To encode using GPU (for SkipThoughts) you may want to use
<code>THEANO_FLAGS=device=gpu python encode_qa_and_text.py</code>

----

### Modified Memory Networks
Answer questions using a modified version of the End-To-End Memory Network [arXiv](https://arxiv.org/abs/1503.08895). The modifications include use of a fixed word embedding layer along with a shared linear projection, and the ability to pick one among multiple-choice multi-word answers. The memory network supports answering in all sources. The main options to run this program are:
(i) story sources: split_plot, subtitle, script;
(ii) number of memory layers (although this did not affect performance much); and
(iii) training parameters: batch size, learning rate, #epochs.


----

### Requirements (in one place)

- Word2Vec: Python installation using <code>pip install word2vec</code>
- SkipThoughts: [Github repo](https://github.com/ryankiros/skip-thoughts)
- Theano: [Github repo](https://github.com/Theano/Theano), tested on some versions of Theano-0.7 and 0.8.
- scikit-learn (PCA)
- python-gflags
- optparse
- nltk
- scipy
- numpy
- Tensorflow
- keras

Reimple
