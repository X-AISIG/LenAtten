from generate_glove_embedding import GloVeEmbedder
from generate_glove_embedding import process_special_token
from data_util.data import Vocab
from data_util import config

print(process_special_token('[PAD]', dim=50))

a = Vocab(config.vocab_path, config.vocab_size)
for i in range(4):
    print(a.id2word(i))

# a = GloVeEmbedder(dim=50)

# print(a.get_embedding("hello"))

# print(a.get_embedding("kkjffkuyjk"))

# print(a.get_embedding("kkjffkuyjk", append_list=True))

# print(a.get_embedding("kkjffkuyjk"))