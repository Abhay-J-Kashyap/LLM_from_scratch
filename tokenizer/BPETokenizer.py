import regex


class BPETokenizer:

    def __init__(self, vocab, merges, pattern):
        self.vocab = vocab
        self.merges = merges
        self.pattern = regex.compile(pattern)

        # ID -> token
        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

    def split_text(self, text):
        return self.pattern.findall(text)

    def get_best_pair(self, tokens):
        best_pair = None
        best_rank = float("inf")

        for pair in zip(tokens, tokens[1:]):
            rank = self.merges.get(pair)

            if rank is not None and rank < best_rank:
                best_pair = pair
                best_rank = rank

        return best_pair

    def merge_pair(self, tokens, pair):
        new_tokens = []
        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_tokens.append(
                    tokens[i] + tokens[i + 1]
                )
                i += 2

            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def encode_chunk(self, chunk):
        # Start from UTF-8 bytes
        tokens = [
            bytes([b])
            for b in chunk.encode("utf-8")
        ]

        # Apply learned BPE merges
        while len(tokens) > 1:

            best_pair = self.get_best_pair(tokens)

            if best_pair is None:
                break

            tokens = self.merge_pair(
                tokens,
                best_pair
            )

        # Convert final byte pieces -> IDs
        return [
            self.vocab[token]
            for token in tokens
        ]

    def encode(self, text):
        ids = []

        for chunk in self.split_text(text):
            ids.extend(
                self.encode_chunk(chunk)
            )

        return ids

    def decode(self, ids):
        byte_data = b"".join(
            self.id_to_token[token_id]
            for token_id in ids
        )

        return byte_data.decode(
            "utf-8",
            errors="replace"
        )