from os.path import exists
from tqdm import tqdm
import bcolz
import numpy as np
import pickle

def preprocess_existence(*args):
    for path in args:
        if not exists(path):
            return False
    return True

def process_special_token(word, dim):
    PAD_TOKEN = '[PAD]'
    UNKNOWN_TOKEN = '[UNK]'
    START_DECODING = '[START]'
    STOP_DECODING = '[STOP]'
    special_tokens = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]
    path = f'/home/fltsm/word-embedding/special.tokens.{dim}d.pkl'
    tokens = {}
    if preprocess_existence(path):
        tokens = pickle.load(open(path, 'rb'))
    else:
        for i in special_tokens:
            tokens[i] = np.random.uniform(low=-1, high=1, size=(dim,))
        pickle.dump(tokens, open(path, 'wb'))
    return tokens[word]

class GloVeEmbedder(object):
    """
    A GloVe Embedder
    Load preprocessed or do preprocessing according to path and dim
    """
    def __init__(self, path="/home/fltsm/word-embedding", dim=50):
        if dim not in [50, 100, 200, 300]:
            raise AttributeError 
        self.dim = dim
        self.source_txt = f'{path}/glove.6B.{dim}d.txt'
        self.vectors_list_path = f'{path}/out.6B.{dim}.dat'
        self.words_list_path = f'{path}/out.6B.{dim}_words.pkl'
        self.ids_list_path = f'{path}/out.6B.{dim}_idx.pkl'
        self.word_list = []
        self.idx = 0
        self.word2idx = {}
        self.vectors = np.array([])
        self.glove = {}
        if preprocess_existence(self.source_txt, self.vectors_list_path, self.words_list_path, self.ids_list_path):
            print("Preprocess Detected")
            self.vectors = bcolz.open(self.vectors_list_path)[:]
            self.word_list = pickle.load(open(self.words_list_path, 'rb'))
            self.word2idx = pickle.load(open(self.ids_list_path, 'rb'))
            self.idx = len(self.vectors)
        else:
            print(f"Processing {dim}d GloVe Embedding")
            self.vectors = bcolz.carray(np.zeros(1), rootdir=self.vectors_list_path, mode='w')
            with open(self.source_txt, 'rb') as f:
                for _, l in enumerate(tqdm(f)):
                    line = l.decode().split()
                    word = line[0]
                    self.word_list.append(word)
                    self.word2idx[word] = self.idx
                    self.idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    self.vectors.append(vect)
            self.dump_embedding()
        self.build_glove_dict()

    def build_glove_dict(self):
        self.glove = {w: self.vectors[self.word2idx[w]] for w in self.word_list}

    def dump_embedding(self, source=True):
        """
        dump the object variable to pickel and bcolz file
        args:
            source: when dumping data processed from original txt, use True (default)
                    when dumping data which is using an appended list, use False
        """
        if source:
            self.vectors = bcolz.carray(self.vectors[1:].reshape((self.idx, self.dim)), rootdir=self.vectors_list_path, mode='w')
        else:
            self.vectors = bcolz.carray(self.vectors[:].reshape((self.idx, self.dim)), rootdir=self.vectors_list_path, mode='w')
        self.vectors.flush()
        pickle.dump(self.word_list, open(self.words_list_path, 'wb'))
        pickle.dump(self.word2idx, open(self.ids_list_path, 'wb'))

    def get_embedding(self, word, append_list=False):
        """
        get embedding list according to given word
        args:
            word: the word you want to get embedding
            append_list: True if you want to append unknown word to the GloVe list
        """
        embedding = []
        try:
            embedding = self.glove[word]
        except KeyError:
            # print("Word %s not exists in GloVe, create a %d random vector for it" % (word, self.dim))
            embedding = np.random.uniform(low=-1, high=1, size=(self.dim,)).astype(np.float)
            if append_list:
                # print("Save random vector to GloVe list")
                self.word_list.append(word)
                self.word2idx[word] = self.idx
                self.idx += 1
                self.vectors = bcolz.carray(self.vectors, rootdir=self.vectors_list_path, mode='w')
                # self.vectors = np.append(self.vectors, [embedding], axis=0)
                self.vectors.append(embedding)
                self.build_glove_dict()
        return embedding
