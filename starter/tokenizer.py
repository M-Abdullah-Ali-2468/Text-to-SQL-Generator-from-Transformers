import json
import sentencepiece as spm
from data_prep import MAX_COLS

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL = ["<sep>"] + [f"<c{i}>" for i in range(MAX_COLS)]


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_tokenizer(train_path="train_pairs.jsonl", vocab_size=8000):
    pairs = read_pairs(train_path)

    with open("spm_corpus.txt", "w", encoding="utf-8") as f:
        for p in pairs:  # ONE shared vocabulary for source and target, as in the original Transformer
            f.write(p["src"] + "\n")
            f.write(p["tgt"] + "\n")

    spm.SentencePieceTrainer.train(
        input="spm_corpus.txt",
        model_prefix="sql_sp",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=SPECIAL,  # never split <sep>, <c0>, ...
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    return spm.SentencePieceProcessor(model_file="sql_sp.model")


if __name__ == "__main__":
    sp = train_tokenizer()

    p = read_pairs("dev_pairs.jsonl")[0]

    print(sp.encode(p["src"], out_type=str))
    print(sp.encode(p["tgt"], out_type=str))
    print(sp.decode(sp.encode(p["tgt"])) == p["tgt"])