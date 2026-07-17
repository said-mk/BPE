import unicodedata

def get_pairs(ids, counts=None):
    # count adjacent id pairs in the sequence

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pairs(ids, pair, idx:int):
    # merge the specified pair and return a new id list

    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) -1 and ids[i] == pair[0] and  ids[i+1] == pair[1] :
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
        
    return new_ids

def replace_control_characters(s:str):
    # replace control characters with unicode escape sequences

    chars =[]
    for ch in s:
        if unicodedata.category(ch)[0] != "C" :
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")

    return "".join(chars)

def render_token(token:bytes):
    # decode a token and clean control characters

    s = token.decode("utf-8", errors ="replace")
    s = replace_control_characters(s)
    return s

# Tokenizer base class
class Tokenizer:
    # BPE tokenizer base class with abstract train, encode, and decode methods
    def __init__(self):
        # initialize merges, vocab, pattern, and special tokens

        self.merges = {} # {(idx0, idx1): idx, ...}
        self.pattern = "" 
        self.special_tokens = {} # {token: idx, ...} e.g. {'<|endoftext|>': 100257}
        self.vocab  = self._build_vocab() # {idx: token, ...}

    def train(self, text, vocab_size, verbose=False):
        # abstract training method
        raise NotImplementedError
    
    def encode(self, text):
        # abstract encode method
        raise NotImplementedError
    
    def decode(self, ids):
        # abstract decode method
        raise NotImplementedError

    def _build_vocab(self):
        # build full byte-pair vocab from base bytes and merges
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (a0, a1), idx in self.merges.items():
            vocab[idx] = vocab[a0] + vocab[a1]
        for special, idx in self.special_tokens.items():
            vocab[idx]= special.encode("utf-8", errors="replace")
        if hasattr(self, 'inverse_special_tokens'):
            self.inverse_special_tokens.clear()
            self.inverse_special_tokens.update({v: k for k, v in self.special_tokens.items()})
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        model_file = file_prefix +".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("Byte pair encoding model file V1\n")
            f.write(f"{self.pattern}\n")
            f.write (f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx0, idx1 in self.merges:
                f.write(f"{idx0} {idx1}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pairs for pairs, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for  idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}  {s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        # load merges and special tokens from a model file
        assert model_file.endswith(".model"), "Model file should have .model extension"
        merges = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "Byte pair encoding model file V1", "Unsupported model version"
            self.pattern = f.readline().strip()
            num_special_tokens = int(f.readline().strip())
            for _ in range(num_special_tokens):
                special, special_idx = f.readline().strip().split()
                self.special_tokens[special] = int(special_idx)
            for line in f:
                idx0, idx1 = map(int, line.strip().split())
                merges[(idx0, idx1)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()
