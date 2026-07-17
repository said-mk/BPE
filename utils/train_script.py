import os
import sys

# Add project root to sys.path so that 'bpe' package can be imported
# regardless of whether the script is run from the root directory or utils/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bpe import BPETokenizer

def main():
    # Resolve the path to the essay file
    essay_path = os.path.join(project_root, "test", "pg_essays.txt")
    if not os.path.exists(essay_path):
        print(f"Error: {essay_path} not found.")
        sys.exit(1)

    with open(essay_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text)} characters of text from {essay_path}.")
    
    # We train for 512 vocab size (256 merges).
    vocab_size = 512
    print(f"Training BPETokenizer with vocab_size={vocab_size}...")
    
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size, verbose=True)
    
    # Save the output files in the current working directory
    model_prefix = "pg_bpe_512"
    tokenizer.save(model_prefix)
    print(f"\nModel successfully saved to:")
    print(f"  - {os.path.abspath(model_prefix + '.model')}")
    print(f"  - {os.path.abspath(model_prefix + '.vocab')}")

if __name__ == "__main__":
    main()
