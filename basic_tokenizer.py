from .base import Tokenizer, get_pairs, merge_pairs

class minBPE(Tokenizer):
     # a tiny BPE implementation for demonstration and learning
     # we start from raw bytes and iteratively absorb the most common pairs
     def __init__(self):
          super().__init__()

     def train(self, text, vocab_size, verbose=False):
          assert vocab_size >= 256, "vocabulary size must be >= 256"
          num_merges = vocab_size - 256

          # work in raw UTF-8 bytes so we can learn byte-pair merges directly
          text_bytes = text.encode("utf-8")
          ids = list(text_bytes)

          merges = {}
          vocab = {idx: bytes([idx]) for idx in range(256)}

          for i in range(num_merges):
               # compute all adjacent pairs and pick the most frequent
               stats = get_pairs(ids)
               if not stats:
                    break
               pair = max(stats, key= stats.get)
               idx = 256 + i

               # replace every occurrence of that pair with a new token id
               ids = merge_pairs(ids, pair, idx)
               merges[pair] = idx
               vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

               if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

          # final model consists of the merge table plus the reconstructed vocab
          self.merges = merges
          self.vocab  = vocab

     def encode(self, text):
          # encode text by repeatedly applying the learned merge rules
          text_bytes = text.encode("utf-8")
          ids = list(text_bytes)

          while len(ids) >= 2:
               stats = get_pairs(ids)
               # choose the next pair that exists in the learned merge order
               pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
               if pair not in self.merges:
                    break
               idx = self.merges[pair]
               ids = merge_pairs(ids, pair, idx)

          return ids

     def decode(self, ids):
          # decode token ids back to bytes, then to text
          text_bytes = b"".join(self.vocab[idx] for idx in ids)
          text = text_bytes.decode("utf-8", errors="replace")
          return text
