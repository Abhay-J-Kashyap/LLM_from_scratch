"""
build_tokenizer_corpus.py

Streams a proportional sample from the five sources cosmo2-tokenizer was
trained on (FineWeb-Edu 70% / Cosmopedia v2 15% / StarCoderData 8% /
OpenWebMath 5% / StackOverflow 2%), caches each source's raw text locally
as JSONL, then trains a byte-level BPE tokenizer on the combined sample
using the GPT-4 pretokenizer regex.

Prereqs:
    pip install datasets tokenizers huggingface_hub

StarCoderData is gated: before running this script, accept the terms at
https://huggingface.co/datasets/bigcode/starcoderdata and run:
    huggingface-cli login
"""

import json
import os

from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Define the source mix. Change `fraction` values (must sum to 1.0) or
#    swap `data_dir` to a different StarCoderData language if you want a
#    different code/math/prose balance than cosmo2's.
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "name": "fineweb_edu",
        "fraction": 0.70,
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "text_field": "text",
    },
    {
        "name": "cosmopedia_v2",
        "fraction": 0.15,
        "repo_id": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "text_field": "text",
    },
    {
        "name": "starcoderdata_python",
        "fraction": 0.08,
        "repo_id": "bigcode/starcoderdata",
        "data_dir": "python",       # gated - see docstring above
        "text_field": "content",
    },
    {
        "name": "openwebmath",
        "fraction": 0.05,
        "repo_id": "open-web-math/open-web-math",
        "config": None,
        "text_field": "text",
    },
    {
        "name": "stackoverflow",
        "fraction": 0.02,
        "repo_id": "mikex86/stackoverflow-posts",
        "config": None,
        "text_field": "Body",       # unfiltered: includes both questions and
                                     # answers, all PostTypeIds. Fine for
                                     # tokenizer training; filter by
                                     # PostTypeId in (1, 2) if you want only
                                     # actual Q&A text for real pretraining.
    },
]

TOTAL_DOCS = 1_000_000   # matches cosmo2-tokenizer's own "1M samples" scale
OUTPUT_DIR = "tokenizer_corpus_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cache_source(source, n_docs):
    """Stream n_docs documents from one source and cache them as JSONL.
    Safe to re-run: skips sources that are already cached on disk."""
    out_path = os.path.join(OUTPUT_DIR, f"{source['name']}.jsonl")
    if os.path.exists(out_path):
        print(f"[{source['name']}] cache already exists, skipping")
        return out_path

    load_kwargs = {"split": "train", "streaming": True}
    if source.get("config"):
        load_kwargs["name"] = source["config"]
    if source.get("data_dir"):
        load_kwargs["data_dir"] = source["data_dir"]

    ds = load_dataset(source["repo_id"], **load_kwargs)

    print(f"[{source['name']}] streaming {n_docs:,} docs -> {out_path}")
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            if written >= n_docs:
                break
            text = example.get(source["text_field"])
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
    print(f"[{source['name']}] wrote {written:,} docs")
    return out_path


def build_all_caches():
    paths = {}
    for source in SOURCES:
        n_docs = int(TOTAL_DOCS * source["fraction"])
        paths[source["name"]] = cache_source(source, n_docs)
    return paths


def corpus_iterator(paths, batch_size=1000):
    """Yields batches of raw text strings, pulled round-robin across sources.
    Note: BPE training counts pair frequencies over the whole corpus before
    merging, so the *final trained vocab* doesn't actually depend on this
    ordering - round-robin is just so progress/logging isn't dominated by
    whichever source happens to be read first."""
    file_handles = {name: open(p, "r", encoding="utf-8") for name, p in paths.items()}
    batch = []
    exhausted = set()
    while len(exhausted) < len(file_handles):
        for name, fh in file_handles.items():
            if name in exhausted:
                continue
            line = fh.readline()
            if not line:
                exhausted.add(name)
                continue
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
    for fh in file_handles.values():
        fh.close()


if __name__ == "__main__":
    paths = build_all_caches()

    # ---- 2. Train the tokenizer on the combined sample --------------------
    from tokenizers import Tokenizer, Regex, pre_tokenizers, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    GPT4_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
    )

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(GPT4_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=32000,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(corpus_iterator(paths), trainer=trainer)
    tokenizer.save("my_tokenizer.json")
    print("Done - tokenizer saved to my_tokenizer.json")