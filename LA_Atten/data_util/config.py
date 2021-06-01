# Anotated Gigawords:
root = "/home/dataset"
train_data_path = 	root + "data/chunked/train/train_*"
valid_data_path = 	root + "data/chunked/valid/valid_*"
test_data_path = 	root + "data/chunked/test/test_*"
vocab_path = 		root + "data/vocab"
save_model_path = root + "data/saved_models/Attn"

# Hyperparameters
vocab_size = 100000                         # Take most frequent vocab_size words. 
emb_dim = 300                               # Dimension of Word Embedding.
num_of_predefined_words = 4                 # [PAD], [UNK], [START], [STOP]
batch_size = 200                            # mini-batch size.
hidden_dim = 256                            # number of units for LSTM.
div_factor = 1000                           # Divide remain length by 1000 to adjust the scope.
lr = 0.001                                  # learning rate.
min_dec_steps = 3

max_enc_steps = 55		                    # 99% of the articles are within length 55.
max_dec_steps = 30		                    # 99% of the titles are within length 15, 
max_word_len = 250                          # \Aleph used in LA.

# Eval settings:
beam_size = 4                               # Beam search size.

# RL 
# -------------
rl_reward_ratio = 10                        # divide rl reward by 10 to match ROUGE score's scope.
rl_len_reward_diff = 4                      # reward the RL if len_diff is not bigger than 70.
eps = 1e-12
eps_denominator = 1e-11
# -------------

rand_unif_init_mag = 0.02                   
trunc_norm_init_std = 1e-4

max_iterations = 50000                      # Total iteration number.
save_model_checkpoint = 10000               # Every 10000 iterations, saves model parameters and infos.
show_train_message = 1000                   # Every 1000 iterations, display training inforation.
intra_encoder = True                        # Use intra_encoder Attention.
intra_decoder = True                        # Use intra_decoder Attention.
print_eval_msg = False                      # Save evaluation message to files.
DEBUG = False                               # For Debug.
glove = True                                # Use glove
cnndm = False
gigawords = True
len_attn_visualization = False
count_space = False
