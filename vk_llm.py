import struct
import sys
import subprocess
import time
import random
from pathlib import Path

from tokenizers import Tokenizer, AddedToken, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "bin" / "main.exe"
PRUNED_VOCAB = ROOT / "pruned-vocab"

GGUF_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GGUF_SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
               5: ("i", 4), 6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


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


class _GgufReader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected end of gguf")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def skip_string(self):
        self.f.seek(self.u64(), 1)

    def skip_value(self, vtype):
        if vtype == 8:
            self.skip_string()
        elif vtype == 9:
            elem = self.u32()
            count = self.u64()
            if elem == 8:
                for _ in range(count):
                    self.skip_string()
            else:
                self.f.seek(count * GGUF_SCALAR[elem][1], 1)
        else:
            self.f.seek(GGUF_SCALAR[vtype][1], 1)

    def str_array(self):
        self.u32()
        count = self.u64()
        return [self.string() for _ in range(count)]

    def int_array(self):
        elem = self.u32()
        count = self.u64()
        fmt, size = GGUF_SCALAR[elem]
        return list(struct.unpack("<%d%s" % (count, fmt), self.read(count * size)))


def _gguf_tokenizer_meta(path):
    with open(path, "rb") as f:
        r = _GgufReader(f)
        if r.read(4) != b"GGUF":
            raise RuntimeError(f"not a gguf file: {path}")
        r.u32()
        r.u64()
        n_kv = r.u64()
        tokens = merges = token_type = None
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            if key == "tokenizer.ggml.tokens":
                tokens = r.str_array()
            elif key == "tokenizer.ggml.merges":
                merges = r.str_array()
            elif key == "tokenizer.ggml.token_type":
                token_type = r.int_array()
            else:
                r.skip_value(vtype)
    return tokens, merges, token_type


def _tokenizer_from_parts(tokens, merges, token_type):
    if not tokens:
        raise RuntimeError("tokenizer has no vocab")
    vocab = {tok: i for i, tok in enumerate(tokens)}
    pairs = []
    for m in merges or []:
        parts = m.split(" ", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    tokenizer = Tokenizer(BPE(vocab, pairs, unk_token=None))
    tokenizer.pre_tokenizer = Sequence([
        Split(Regex(GGUF_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tokenizer.decoder = ByteLevelDecoder()
    if token_type:
        specials = [AddedToken(tok, special=True) for tok, t in zip(tokens, token_type) if t in (2, 3, 4)]
        if specials:
            tokenizer.add_tokens(specials)
    return tokenizer


def _tokenizer_from_gguf(path):
    return _tokenizer_from_parts(*_gguf_tokenizer_meta(path))


def _hqm_meta(path):
    with open(path, "rb") as f:
        r = _GgufReader(f)
        magic = r.read(4)
        if magic[:3] != b"HQM":
            raise RuntimeError(f"not an hqm file: {path}")
        r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kvs = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            if vtype == 8:
                kvs[key] = r.string()
            elif vtype == 12:
                kvs[key] = struct.unpack("<d", r.read(8))[0]
            elif vtype in (4, 7):
                kvs[key] = struct.unpack("<I", r.read(4))[0]
            elif vtype == 10:
                kvs[key] = r.u64()
            else:
                raise RuntimeError(f"unknown hqm kv type {vtype}")
        tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            ndim = r.u32()
            dims = [r.u64() for _ in range(ndim)]
            ttype = r.u32()
            offset = r.u64()
            tensors[name] = (ttype, dims, offset)
        data_offset = kvs["hqm.data_offset"]
    return kvs, tensors, data_offset


def _hqm_read_tensor(path, tensors, data_offset, name):
    ttype, dims, offset = tensors[name]
    size = 1
    for d in dims:
        size *= d
    if ttype == 0:
        nbytes = size * 4
    elif ttype == 1:
        nbytes = size * 2
    elif ttype in (2, 4):
        nbytes = size
    elif ttype == 3:
        nbytes = size // 2
    elif ttype == 5:
        nbytes = size * 4
    else:
        raise RuntimeError(f"unknown hqm tensor type {ttype}")
    with open(path, "rb") as f:
        f.seek(data_offset + offset)
        return f.read(nbytes)


def _read_str_array(raw):
    count = struct.unpack_from("<I", raw, 0)[0]
    pos = 4
    out = []
    for _ in range(count):
        length = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        out.append(raw[pos:pos + length].decode("utf-8", errors="replace"))
        pos += length
    return out


def _tokenizer_from_hqm(path):
    kvs, tensors, data_offset = _hqm_meta(path)
    if "tokenizer.json" in tensors:
        raw = _hqm_read_tensor(path, tensors, data_offset, "tokenizer.json")
        return Tokenizer.from_str(raw.decode("utf-8"))
    if "tokenizer.tokens" in tensors:
        tokens = _read_str_array(_hqm_read_tensor(path, tensors, data_offset, "tokenizer.tokens"))
        merges = None
        if "tokenizer.merges" in tensors:
            merges = _read_str_array(_hqm_read_tensor(path, tensors, data_offset, "tokenizer.merges"))
        token_type = None
        if "tokenizer.token_type" in tensors:
            raw = _hqm_read_tensor(path, tensors, data_offset, "tokenizer.token_type")
            token_type = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        return _tokenizer_from_parts(tokens, merges, token_type)
    raise RuntimeError(f"hqm has no tokenizer: {path}")


def _read_u32(proc):
    raw = proc.stdout.read(4)
    if len(raw) != 4:
        raise RuntimeError("backend exited unexpectedly")
    return struct.unpack("<I", raw)[0]


def start_llm(weight_dir, max_ctx=32768, max_new_tokens=128, dump_dir=None, dump_layers=0, debug_sampling=False, prune_vocab=False, experts_vram=0, export=True, export_dir=None):
    import shutil
    weight_path = Path(weight_dir).resolve()
    is_gguf = weight_path.is_file() and weight_path.suffix.lower() == ".gguf"
    is_hqm = weight_path.is_file() and weight_path.suffix.lower() == ".hqm"
    model_dir = weight_path.parent if (is_gguf or is_hqm) else weight_path
    do_prune = prune_vocab and not is_hqm

    tokenizer_path = (model_dir / "vocab" / "tokenizer.json") if do_prune else (model_dir / "tokenizer.json")

    if do_prune and not tokenizer_path.exists():
        vocab_dir = model_dir / "vocab"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
            src = PRUNED_VOCAB / name
            if src.exists():
                shutil.copy(src, vocab_dir / name)

    if is_hqm:
        tokenizer = _tokenizer_from_hqm(weight_path)
    elif tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    elif is_gguf:
        tokenizer = _tokenizer_from_gguf(weight_path)
    else:
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    eos = tokenizer.token_to_id("<|im_end|>")
    if eos is None:
        eos = 0

    cmd = [
        str(BACKEND),
        "--weights", str(weight_path),
        "--max-ctx", str(max_ctx),
        "--max-new", str(max_new_tokens),
    ]
    if is_gguf:
        cmd += ["--gguf"]
    if not export:
        cmd += ["--no-export"]
    if export_dir is not None:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        cmd += ["--export-dir", str(Path(export_dir).resolve())]
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        cmd += ["--dump", str(dump_dir), "--dump-layers", str(dump_layers)]
    if debug_sampling:
        cmd += ["--debug-sampling"]
    if do_prune:
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
    llm.weight_dir = model_dir
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
    argv = sys.argv[1:]
    thinking = "--think" in argv
    prune = "--prune" in argv
    export = "--no-export" not in argv
    evram = 0
    export_dir = None
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--experts-vram" and i + 1 < len(argv):
            evram = int(argv[i + 1])
            i += 2
            continue
        if a == "--export-dir" and i + 1 < len(argv):
            export_dir = argv[i + 1]
            i += 2
            continue
        if a not in ("--think", "--prune", "--no-export"):
            args.append(a)
        i += 1
    weight_dir = args[0] if len(args) > 0 else "model/Qwen3.5-9B"
    max_ctx = int(args[1]) if len(args) > 1 else 16384
    text = args[2] if len(args) > 2 else None
    if text is None:
        return

    llm = start_llm(weight_dir, max_ctx=max_ctx, max_new_tokens=16384, prune_vocab=prune, experts_vram=evram, export=export, export_dir=export_dir)
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
