import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from beam_search import *
from rouge import Rouge
from tqdm import tqdm
import argparse
import nltk
import sinu_embedding
from generate_glove_embedding import GloVeEmbedder

# For displaying length attention:
import sys
sys.path.append('/home/fltsm/Summarization')
import EvaluationToolkit.heatmap as heatmap

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def compute_var_character(decoded_sents, ref_sents):
    total = 0.0

    for i in range(len(decoded_sents)):    
        model_generate = nltk.word_tokenize(decoded_sents[i])
        gt = nltk.word_tokenize(ref_sents[i])

        if config.count_space:
            model_generated_len = len(' '.join(model_generate))
            gt_len = len(' '.join(gt))
        else:
            model_generated_len = len(''.join(model_generate))
            gt_len = len(''.join(gt))

        total += (model_generated_len - gt_len) ** 2

    var = total / len(decoded_sents)
    return var

class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        """
        Load model
        """
        self.model = Model()
        # Apply GloVe embeds to model embedding
        # -----------------------------------
        if config.glove:
            print('Using glove.')
            embedder = GloVeEmbedder(dim=config.emb_dim)
            glove = np.array([[]])

            for index in range(config.vocab_size):
                if index < config.num_of_predefined_words:
                    # Generate vector for token
                    np.append(glove, np.random.random(config.emb_dim))
                else:
                    # Fetch vector for words
                    # print(index, self.vocab.id2word(index))
                    np.append(glove, embedder.get_embedding(self.vocab.id2word(index)))
            
            # Apply GloVe embeds to model embedding
            self.model.embeds.from_pretrained(T.from_numpy(glove), freeze=True)

        # -----------------------------------

        self.model = get_cuda(self.model)

        # Loading model:
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])


    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        file_prefix = self.opt.task + "_" + str(config.max_word_len) + "_"
        
        filename = "Atten_AEG_" + file_prefix + loadfile.split(".")[0] + ".txt"
        print(filename)

        with open(os.path.join(filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: "+article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, print_sents=False):

        self.setup_valid()
        # get first batch data
        batch = self.batcher.next_batch()
        
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)

        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()

        # Display length attention table:
        if config.len_attn_visualization:
            count = 0

        while batch is not None:
            # load articles:
            articles, artile_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e, character_original_article_words = get_enc_data(batch)
            
            # Get target character lens:
            target_lens = get_dec_data(batch)
            target_lens = target_lens[-2]
            
            
            with T.autograd.no_grad():
                # get embeddings
                articles = self.model.embeds(articles)

                # send embeddings into encoder and get result
                enc_out, enc_hidden = self.model.encoder(articles, character_original_article_words, artile_lens)


            #-----------------------Summarization----------------------------------------------------
            # OUR Model:
            # Pass target length to beam search.
            with T.autograd.no_grad():
                pred_ids, len_attns, remain_lens = beam_search(enc_hidden,
                                                               enc_out,
                                                               enc_padding_mask,
                                                               ct_e,
                                                               extra_zeros,
                                                               enc_batch_extend_vocab,
                                                               self.model,
                                                               start_id,
                                                               end_id,
                                                               unk_id,
                                                               target_lens,
                                                               self.vocab,
                                                               batch.art_oovs)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                
                decoded_sents.append(decoded_words)

                # draw length attention hotplot
                if config.len_attn_visualization:
                    heatmap.draw_attention_heatmap(len_attns, decoded_words, remain_lens, "image/", "C" + str(config.max_word_len) + "_" + str(count), (12, 8))
                    count += 1

                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
                
            batch = self.batcher.next_batch()

        load_file = self.opt.load_model

        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents, avg = True)
        
        # We use rouge-l f score to evaluate our model.
        rouge_l = scores["rouge-l"]["f"]
        rouge_1 = scores["rouge-1"]["f"]
        rouge_2 = scores["rouge-2"]["f"]

        print(load_file, "rouge_l:", "%.4f" % rouge_l)
        print(load_file, "rouge_1:", "%.4f" % rouge_1)
        print(load_file, "rouge_2:", "%.4f" % rouge_2)
        
        character_var = compute_var_character(decoded_sents, ref_sents)
        
        print(load_file, 'length variance(Character Level):', character_var, 'After divide by 1000:', character_var/1000)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate","test"])
    parser.add_argument("--start_from", type=str, default="0010000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--prefix', type=str, help='Set the prefix of storage dir.')
    parser.add_argument('--len_attn_visual', action='store_true', help='Turn on length visualization unit')
    
    opt = parser.parse_args()
    config.len_attn_visualization = opt.len_attn_visual

    if opt.task == "validate":
        # Get all the model saving files.
        if opt.prefix:
            save_path = config.save_model_path + "/" + opt.prefix
            config.save_model_path = save_path

        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        
        # find the models that are required to be validated:
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]

        # Evaluate the specified models.
        for f in tqdm(saved_models):
            opt.load_model = f
            # Change settings when display length attention hot plot.
            if config.len_attn_visualization:
                print('Display length attention table mode on.')
                config.beam_size = 1
                eval_processor = Evaluate(config.valid_data_path, opt, batch_size = 1)
            else:    
                eval_processor = Evaluate(config.valid_data_path, opt)

            eval_processor.evaluate_batch(config.print_eval_msg)
    else:   
        # test
        if opt.prefix:
            save_path = config.save_model_path + "/" + opt.prefix
            config.save_model_path = save_path
            opt.load_model = opt.start_from

        for _ in tqdm(range(1)):
            eval_processor = Evaluate(config.test_data_path, opt)
            eval_processor.evaluate_batch(config.print_eval_msg)
