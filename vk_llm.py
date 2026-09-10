import struct
import sys
import subprocess
import time
import random
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "bin" / "main.exe"
PRUNED_VOCAB = ROOT / "pruned-vocab"

is_sampling = False
temperature = 0.6
rep_penalty = 1.05
penalty_len = 64
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 0.0
seed = None


class LLM:
    pass


def _read_u32(proc):
    raw = proc.stdout.read(4)
    if len(raw) != 4:
        raise RuntimeError("backend exited unexpectedly")
    return struct.unpack("<I", raw)[0]


def start_llm(weight_dir, max_ctx=32768, max_new_tokens=128, dump_dir=None, dump_layers=0, debug_sampling=False, prune_vocab=False, experts_vram=0):
    import shutil
    weight_dir = Path(weight_dir).resolve()
    tokenizer_path = (weight_dir / "vocab" / "tokenizer.json") if prune_vocab else (weight_dir / "tokenizer.json")

    if prune_vocab and not tokenizer_path.exists():
        vocab_dir = weight_dir / "vocab"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
            src = PRUNED_VOCAB / name
            if src.exists():
                shutil.copy(src, vocab_dir / name)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|im_end|>")
    if eos is None:
        eos = 0

    cmd = [
        str(BACKEND),
        "--weights", str(weight_dir),
        "--max-ctx", str(max_ctx),
        "--max-new", str(max_new_tokens),
    ]
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        cmd += ["--dump", str(dump_dir), "--dump-layers", str(dump_layers)]
    if debug_sampling:
        cmd += ["--debug-sampling"]
    if prune_vocab:
        cmd += ["--prune"]
    if experts_vram > 0:
        cmd += ["--experts-vram", str(experts_vram)]

    proc = subprocess.Popen(
        cmd,
        cwd=str(BACKEND.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    llm = LLM()
    llm.proc = proc
    llm.weight_dir = weight_dir
    llm.max_ctx = max_ctx
    llm.max_new_tokens = max_new_tokens
    llm.tokenizer = tokenizer
    llm.eos = eos
    return llm


def apply_chat_template(llm, messages, system="", enable_thinking=True):
    bos = "<|im_start|>"
    eos = "<|im_end|>"
    parts = []
    if (system != ""):
        parts.append("<|im_start|>system\n")
        parts.append(system)
        parts.append(eos+"\n")
    for msg in messages:
        parts.append(f"{bos}{msg['role']}\n{msg['content']}{eos}\n")
    parts.append(f"{bos}assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return llm.tokenizer.encode("".join(parts), add_special_tokens=False).ids


def tokenize(llm, text, thinking):
    return apply_chat_template(llm, [{"role": "user", "content": text}], system="You are a helpful assistant.", enable_thinking=thinking)


def detokenize(llm, token_ids):
    return llm.tokenizer.decode(token_ids)


def _stream_ids(llm, token_ids):
    if not token_ids:
        return
    n = len(token_ids)
    if n >= llm.max_ctx:
        raise ValueError(f"prompt length {n} exceeds max_ctx {llm.max_ctx}")
    llm.proc.stdin.write(struct.pack("<I", n))
    if is_sampling:
        s = seed if seed is not None else random.getrandbits(32)
        if s == 0:
            s = 1
        header = (temperature, rep_penalty, penalty_len, top_k, top_p, min_p, presence_penalty, s)
    else:
        header = (0.0, 1.0, 0, 0, 1.0, 0.0, 0.0, 1)
    llm.proc.stdin.write(struct.pack("<ffIIfffI", *header))
    llm.proc.stdin.write(struct.pack("<%dI" % n, *token_ids))
    llm.proc.stdin.flush()
    while True:
        tok = _read_u32(llm.proc)
        if tok == 0xFFFFFFFF:
            return
        yield tok


def generate(llm, token_ids):
    return list(_stream_ids(llm, token_ids))


def generate_stream(llm, token_ids):
    ids = []
    prev = ""
    for tok in _stream_ids(llm, token_ids):
        ids.append(tok)
        text = llm.tokenizer.decode(ids)
        if len(text) > len(prev):
            yield text[len(prev):]
        prev = text


def close(llm):
    try:
        llm.proc.stdin.write(struct.pack("<I", 0))
        llm.proc.stdin.flush()
    except Exception:
        pass
    try:
        llm.proc.stdin.close()
    except Exception:
        pass
    llm.proc.wait(timeout=10)
    llm.proc.terminate()


def _main():
    args = [a for a in sys.argv[1:] if a not in ("--think", "--prune") and not a.startswith("--experts-vram")]
    thinking = "--think" in sys.argv[1:]
    prune = "--prune" in sys.argv[1:]
    evram = 0
    for i, a in enumerate(sys.argv[1:]):
        if a == "--experts-vram" and i + 2 < len(sys.argv):
            evram = int(sys.argv[i + 2])
    weight_dir = args[0] if len(args) > 0 else "model/Qwen3.5-9B"
    max_ctx = int(args[1]) if len(args) > 1 else 16384
    text = args[2] if len(args) > 2 else None
    if text is None:
        return

    llm = start_llm(weight_dir, max_ctx=max_ctx, max_new_tokens=16384, prune_vocab=prune, experts_vram=evram)
    ids = tokenize(llm, text, thinking)
    decoded_ids = []
    prev = ""
    timestamps = []
    for tok in _stream_ids(llm, ids):
        timestamps.append(time.perf_counter())
        decoded_ids.append(tok)
        text = llm.tokenizer.decode(decoded_ids)
        if len(text) > len(prev):
            print(text[len(prev):], end="", flush=True)
        prev = text
    print("")
    if len(timestamps) > 4:
        n = len(timestamps) - 4
        elapsed = timestamps[-1] - timestamps[3]
        if elapsed > 0:
            print(f"{len(timestamps)} tokens, {n / elapsed:.1f} tokens/s")
    close(llm)


if __name__ == "__main__":
    _main()
