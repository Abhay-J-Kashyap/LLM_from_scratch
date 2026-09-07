import regex
import json

GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

reg = regex.compile(GPT4_PATTERN)


class BPETrainer:

    def split_text(self, text):
        return reg.findall(text)

    def get_pair_counts(self, tokens):
        counts = {}

        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def merge_pair(self, tokens, best_pair):
        new_tokens = []
        j = 0

        while j < len(tokens):
            if (
                j < len(tokens) - 1
                and (tokens[j], tokens[j + 1]) == best_pair
            ):
                new_tokens.append(tokens[j] + tokens[j + 1])
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1

        return new_tokens

    def train(self, text, vocab_size, verbose=False):

        #pre-tokenization
        chunks = self.split_text(text)

        tokenized_chunks = []

        for chunk in chunks:
            byte_tokens = [bytes([b]) for b in chunk.encode("utf-8")]
            tokenized_chunks.append(byte_tokens)

        vocab = {bytes([i]) for i in range(256)}

        merges = {}

        while len(vocab) < vocab_size:

            # Find the best pair across all chunks
            global_counts = {}

            for tokens in tokenized_chunks:
                counts = self.get_pair_counts(tokens)

                for pair, count in counts.items():
                    global_counts[pair] = (global_counts.get(pair, 0) + count)

            if not global_counts:
                break

            best_pair = max(global_counts, key=global_counts.get)

            merges[best_pair] = len(merges)

            # Merge this pair independently inside each chunk
            new_chunks = []

            for tokens in tokenized_chunks:
              merged_tokens = self.merge_pair(tokens,best_pair)
              new_chunks.append(merged_tokens)
            
            tokenized_chunks = new_chunks

            # Add the newly created token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

            if verbose:
                print(
                    f"merge {len(merges) - 1}: "
                    f"{best_pair} -> {new_token}"
                )

        return vocab, merges



    def build_vocab(self, merges):
        vocab = {bytes([i]): i for i in range(256)}

        for pair, rank in sorted(merges.items(), key=lambda x: x[1]):
            token = pair[0] + pair[1]
            vocab[token] = 256 + rank

        return vocab

    # =========================================================
    # SAVE TOKENIZER
    # =========================================================

    def save(self, path, merges, vocab):
        """
        Save tokenizer to a JSON file.

        bytes cannot be directly stored in JSON,
        so they are represented as lists of integers.
        """

        data = {
            "merges": [
                {
                    "left": list(pair[0]),
                    "right": list(pair[1]),
                    "rank": rank
                }
                for pair, rank in merges.items()
            ],

            "vocab": [
                {
                    "token": list(token),
                    "id": token_id
                }
                for token, token_id in vocab.items()
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2
            )