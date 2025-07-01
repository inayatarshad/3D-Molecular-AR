# utils/vocab.py
import pickle

class SMILESVocab:
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']

    def build_vocab(self, smiles_list):
        unique_chars = set(''.join(smiles_list))
        all_tokens = self.special_tokens + sorted(unique_chars)

        self.token_to_idx = {ch: idx for idx, ch in enumerate(all_tokens)}
        self.idx_to_token = {idx: ch for ch, idx in self.token_to_idx.items()}

    def load_vocab(self, vocab_file):
        with open(vocab_file, "rb") as f:
            self.token_to_idx = pickle.load(f)
        self.idx_to_token = {idx: ch for ch, idx in self.token_to_idx.items()}

    def get_idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx['<unk>'])

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, smiles):
        tokens = ['<start>'] + list(smiles) + ['<end>']
        return [self.get_idx(token) for token in tokens]

    def decode(self, indices):
        tokens = [self.idx_to_token.get(idx, '<unk>') for idx in indices]
        return ''.join(tokens).replace('<start>', '').replace('<end>', '')
