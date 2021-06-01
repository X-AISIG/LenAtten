# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py
import queue as Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

from . import config
from . import data

import random
random.seed(1234)


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, article, abstract_sentences, vocab):
    """Initializes the Example, performing tokenization and truncation
      to produce the encoder, decoder and target sequences, which are stored in self.
    
    Args:
      article: source text; a string. each token is separated by a single space.
      
      abstract_sentences(GT/Test data): list of strings, one per abstract sentence.
        In each sentence, each token is separated by a single space.
      
      vocab: Vocabulary object
    """
    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)
    # Process the article
    article_words = article.split()
    
    # Truncation
    # Only store the amount of original article words that not exceed max_enc_steps:
    if len(article_words) > config.max_enc_steps:
      article_words = article_words[:config.max_enc_steps]

    # Store the length after truncation(Delete word after max_enc_steps) but before padding
    # self.enc_len: Length of original Text.
    self.enc_len = len(article_words) 
    # MODEL:
    # Need to get original article length and use it to calculate the 
    # Positional Encoding, https://arxiv.org/abs/1706.03762
    self.article_len = self.enc_len

    # Non pointer-generator mode:
    # list of word ids; OOVs are represented by the id for UNK token
    # self.enc_input: word id representation of the original Text.
    # contains: word ids in pre-defined word list + id of UNK; No in article oovs.
    self.enc_input = [vocab.word2id(w) for w in article_words] 

    # Using pointer-generator mode, we need to store some extra info
    # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
    # also store the in-article OOVs words themselves.

    # enc_input_extend_vocab: word id representation of the original Text.
      # contains:  word ids in pre-defined word list + temp ids of in-article OOVs + id of UNK;
    # article_oovs: contain oov words from the article.
    self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

    # Process the abstract/GT
    abstract = ' '.join(abstract_sentences) # string

    abstract_words = abstract.split() # list of strings(Ground Truth)

    # list of word ids; OOVs are represented by the id for UNK token
    # abs_ids: id representation of the untruncated ground truth.
    # contains: word ids in pre-defined word list + id of UNK; No in article oovs.
    abs_ids = [vocab.word2id(w) for w in abstract_words]

    # Get the input sequence for the decoder.
    # In decoding step: we need x^{t-1} to feed into the decoder, which shouldn't contains in-article OOV.
    # self.dec_input: It's the id representation of the ground truth after truncation.
    self.dec_input, _ = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    
    # self.dec_len: # (<SOS> + ground truth).
    self.dec_len = len(self.dec_input)


    # Get a verison of the reference summary
    # where in-article OOVs are represented by their temporary article OOV id
    # Could find in-article OOVs from article_oovs later
    # abs_ids_extend_vocab: word ids for ground truth
    # contains: word ids in original word list + word (temp) ids of in-article oovs + id of UNK

    # Actually, this is the id representation of the ground truth(Before truncation).
    abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

    # Get decoder target sequence
    # A.K.A: y.
    # self.target: It's the id representation of the ground truth after truncation(if necessary).
    _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)
    

    # OUR MODEL:
    # FORMAT: ground truth + <EOS>
    # For current length, need to start from 1!.
    self.target_length = len(self.target)

    # Store the original strings
    self.original_article = article
    self.original_abstract = abstract
    self.original_abstract_sents = abstract_sentences


    # Our character level embedding:
    truncted_gt_summary = abstract_words[:config.max_dec_steps]
    self.original_article_words_character_num = [len(word) for word in article_words]
    self.original_abstract_sents_words_character_num = [len(word) for word in truncted_gt_summary]
    if config.count_space:
      # Count length when computing the length of target summary
      # print('count space in batcher.py')
      self.total_summary_character = len(' '.join(truncted_gt_summary))
    else:
      # print('dont count space in batcher.py')
      # Doesn't count length when computing the length of target summary
      self.total_summary_character = len(''.join(truncted_gt_summary))



  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens,
     return the input sequence for the decoder,
     and the target sequence which we will use to calculate loss.
    
    The sequence will be truncated if it is longer than max_len.
    The input sequence must start with the start_id
    and the target sequence must end with the stop_id (but not if it's been truncated).
    
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    # <s> + ground truth

    target = sequence[:]
    # ground truth

    # if the length of GT is larger than max_decoding step:
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)

    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """
    Pad decoder input and target sequences with pad_id up to max_len.
    For handling batch_size article at the same time.
    """

    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)

    while len(self.target) < max_len:
      self.target.append(pad_id)

    while len(self.original_abstract_sents_words_character_num) < max_len:
      self.original_abstract_sents_words_character_num.append(0)



  def pad_encoder_input(self, max_len, pad_id):
    """
    Pad the encoder input sequence with pad_id up to max_len.
    For handling batch_size article at the same time.
    """

    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
      
    while len(self.enc_input_extend_vocab) < max_len:
      self.enc_input_extend_vocab.append(pad_id)

    while len(self.original_article_words_character_num) < max_len:
      self.original_article_words_character_num.append(0)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, vocab, batch_size):
    """Turns the example_list into a Batch object.
    
    Args:
       example_list: List of Example objects
       vocab: Vocabulary object
       batch_size: Mini batch size.
    """

    self.batch_size = batch_size
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings


  def init_encoder_seq(self, example_list):
    # Determine the maximum length of the encoder input sequence in this batch
    self.max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(self.max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension)
    # for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((self.batch_size, self.max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
    
    # OUR MODEL: Store the length of original article for each example
    self.article_lens = self.enc_lens

    self.enc_padding_mask = np.zeros((self.batch_size, self.max_enc_seq_len), dtype=np.float32)

    
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]

      # This is the original article length.
      self.enc_lens[i] = ex.enc_len

      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    # Determine the max number of in-article OOVs in this batch
    self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
    # Store the in-article OOVs themselves
    self.art_oovs = [ex.article_oovs for ex in example_list]
    # Store the version of the enc_batch that uses the article OOV ids
    self.enc_batch_extend_vocab = np.zeros((self.batch_size, self.max_enc_seq_len), dtype=np.int32)
    for i, ex in enumerate(example_list):
      self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list):
    # Pad the inputs and targets

    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    # NOT USED
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len

      # Not Used
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1
      

  def store_orig_strings(self, example_list):
    # OUR MODEL: Store the length of Ground truth of each example
    self.gt_lens = np.zeros((self.batch_size), dtype=np.int32)
    self.character_summary_total = np.zeros((self.batch_size), dtype=np.int32)

    self.character_summary = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.character_original_article = np.zeros((self.batch_size, self.max_enc_seq_len), dtype=np.int32)

    for i, ex in enumerate(example_list):
      # OUR MODEL:
      # This is the length of the GT
      self.gt_lens[i] = ex.target_length
      
      # Character level embedding:
      # GT summary total length:
      self.character_summary_total[i] = ex.total_summary_character

      # GT summary length:
      # self.character_summary.append(ex.original_abstract_sents_words_character_num)
      self.character_summary[i] = ex.original_abstract_sents_words_character_num

      # self.character_original_article.append(ex.original_article_words_character_num)
      self.character_original_article[i] = ex.original_article_words_character_num

      if config.DEBUG:
        print('*Testing original article character length', self.character_original_article[i])
        print('*Testing total characters of a summary:', self.character_summary_total[i])
        print('*Testing character length of a summary:', self.character_summary[i])

    self.original_articles = [ex.original_article for ex in example_list] # list of lists
    self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher(object):
  """A class to generate minibatches of data.
   Buckets examples together based on length of the encoder sequence.
  """

  BATCH_QUEUE_MAX = 1000 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    """Initialize the batcher. Start threads that process the data into batches.
    
    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set).
      Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 #16 # num threads to fill example queue
      self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
      self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def next_batch(self):
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.info('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())

      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """
    Reads data from data file.
    For each data point (article, abstract), form them into an Example obj, and
    enqueue it into _example_queue.
    """
    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (article, abstract) = next(input_gen) 
        # read the next example from file. article and abstract are both strings.
      except Exception: # if there are no more examples:
        # print("The example generator for this example queue filling thread has exhausted data.")
        
        if self._single_pass:
          # print("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      # FIXME:May contains bug here
      if config.cnndm:
        # print('Using CNNDM')
        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] 
      
      # FIXME:Use the <s> and </s> tags in abstract to get a list of sentences.
      if config.gigawords:
        # print('Using Anotated Gigawords')
        abstract_sentences = [abstract.strip()]

      example = Example(article, abstract_sentences, self._vocab) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.

  def fill_batch_queue(self):
    """
    Takes Examples out of example queue, sorts them by encoder sequence length,
      processes into Batches with each size equal to batch_size,
      and places them in the batch queue.
    
    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self.mode == 'decode':
        # In beam search decode mode:
        # single example repeated in the batch
        ex = self._example_queue.get()
        example_list = [ex for _ in range(self.batch_size)]
        self._batch_queue.put(Batch(example_list, self._vocab, self.batch_size))
      else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []

        for _ in range(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())

        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        
        # Form batches with mini-batch size:
        for i in range(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])

        if not self._single_pass:
          shuffle(batches)

        # Put all batches into the batch queue.  
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

  def watch_threads(self):
    while True:
      tf.logging.info(
        'Bucket queue size: %i, Input queue size: %i',
        self._batch_queue.qsize(), self._example_queue.qsize())

      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    """Generates article and abstract text"""

    while True:
      e = next(example_generator)
      try:
        article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        article_text = article_text.decode()
        abstract_text = abstract_text.decode()
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        #tf.logging.warning('Found an example with empty article text. Skipping it.')
        continue
      else:
        yield (article_text, abstract_text)
