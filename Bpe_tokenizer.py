"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- BpeTokenizer handles an optional regex splitting pattern.
- BpeTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_pairs, merge_pairs


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BPETokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size > 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        chunk_ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        stats = {}
        merges = {}
        vocab = {idx:bytes([idx]) for idx in range(256) }
        
        for i in range(num_merges):
            stats = {}
            for chunk in chunk_ids:
                get_pairs(chunk, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunk_ids = [merge_pairs(chunk, pair, idx) for chunk in chunk_ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        # save class variables
        self.merges = merges # used  in encode
        self.vocab = self._build_vocab() # rebuild vocab to include special tokens

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        bytes_to_decode = []
        for idd in ids:
            if idd in self.vocab:
                bytes_to_decode.append(self.vocab[idd])
            elif idd in self.inverse_special_tokens:
                bytes_to_decode.append(self.inverse_special_tokens[idd].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idd}")
        text_bytes = b"".join(bytes_to_decode)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes:bytes):

        ids = list(text_bytes)

        while len(ids) >= 2:
            stat = get_pairs(ids)
            pair = min(stat, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge_pairs(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids =[]
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    def encode(self, text, allowed_special="none_raise"):

        if allowed_special == 'all':
            special_tokens = self.special_tokens
        elif allowed_special == "none":
            special_tokens = {}
        elif allowed_special == "none_raise":
            special_tokens = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"Encountered text corresponding to disallowed special token {token!r}.")
        elif isinstance(allowed_special, set):
            special_tokens = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} is not valid")
        
        if not special_tokens:
            return self.encode_ordinary(text)
        
        special_token_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        special_chunk_pattern = re.split(special_token_pattern, text)

        ids = []
        for chunk in special_chunk_pattern:
            if chunk in special_tokens:
                ids.append(special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids
        