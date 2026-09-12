import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

import vk_llm

ROOT = vk_llm.ROOT
QUANT_TMP = ROOT / "bin" / "webui_quant.json"
DEFAULT_EXPORT_DIR = ROOT / "exported"
QUANT_CHOICES = [("FP16", "fp16"), ("INT8", "int8"), ("INT4", "int4")]
PRESET_CHOICES = [("custom", "custom")] + QUANT_CHOICES
QUANT_VALUES = {value for _, value in QUANT_CHOICES}
DEFAULT_MAX_CTX = 32768
DEFAULT_PREFILL = 512

_QUANT_BY_VALUE = {4: "int4", 8: "int8", 16: "fp16"}
_TYPE_BY_VALUE = {1: "full_attention", 2: "linear_attention"}

_ENGINE = {"llm": None, "path": None, "kind": None, "info": {}, "lock": threading.Lock()}

_THINK_RE = re.compile(r"(?s)<think>.*?</think>")

_CSS = ".locked { opacity: 0.55; }"


def _model_kind(path):
    p = Path(path)
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".gguf":
            return "gguf"
        if suffix == ".hqm":
            return "hqm"
        if suffix == ".safetensors":
            return "safetensors"
        return "unknown"
    if p.is_dir():
        return "safetensors"
    return "missing"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shard_names(model_dir):
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = _read_json(index)
            names = sorted(set(data.get("weight_map", {}).values()))
            if names:
                return names, "index"
        except Exception:
            pass
    shards = sorted(p.name for p in model_dir.glob("model*.safetensors"))
    return shards, "glob" if shards else "none"


def validate_safetensors(model_dir):
    model_dir = Path(model_dir)
    info = {
        "kind": "safetensors",
        "path": str(model_dir),
        "dir": str(model_dir),
        "ok": False,
        "errors": [],
        "warnings": [],
        "layers": 0,
        "layer_types": [],
        "first_shard": None,
        "name": model_dir.name,
        "experts": 0,
        "max_ctx": DEFAULT_MAX_CTX,
        "max_pos": DEFAULT_MAX_CTX,
    }

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        info["errors"].append("config.json not found")
    else:
        try:
            cfg = _read_json(config_path)
        except Exception as exc:
            info["errors"].append("config.json invalid: %s" % exc)
            cfg = None
        if cfg is not None:
            text = cfg.get("text_config", cfg)
            required = ["hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "layer_types"]
            missing = [k for k in required if k not in text]
            if missing:
                info["errors"].append("config.json missing: " + ", ".join(missing))
            layers = text.get("layer_types", [])
            count = int(text.get("num_hidden_layers", 0) or 0)
            if count and len(layers) != count:
                info["errors"].append("layer_types count %d != num_hidden_layers %d" % (len(layers), count))
            info["layers"] = count
            info["layer_types"] = layers
            max_pos = int(text.get("max_position_embeddings", DEFAULT_MAX_CTX) or DEFAULT_MAX_CTX)
            info["max_ctx"] = max_pos
            info["max_pos"] = max_pos
            info["experts"] = int(text.get("num_experts", 0) or 0)
            info["tied"] = bool(text.get("tie_word_embeddings", False))

    names, source = _shard_names(model_dir)
    info["shard_source"] = source
    info["first_shard"] = names[0] if names else None
    if not names:
        info["errors"].append("no safetensors shards found")
    else:
        missing = {n for n in names if not (model_dir / n).is_file()}
        totals = {}
        for n in names:
            m = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", n)
            if m:
                totals.setdefault(m.group(1), set()).add(int(m.group(3)))
        for prefix, total_set in totals.items():
            total = max(total_set)
            for i in range(1, total + 1):
                candidate = "%s%05d-of-%05d.safetensors" % (prefix, i, total)
                if not (model_dir / candidate).is_file():
                    missing.add(candidate)
        if missing:
            info["errors"].append("missing shards: " + ", ".join(sorted(missing)))

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        if not (model_dir / name).is_file():
            info["warnings"].append("%s not found (prune mode can supply it)" % name)

    info["ok"] = not info["errors"]
    return info


def read_gguf_info(path):
    path = Path(path)
    meta = vk_llm.gguf_meta(path)

    def get(suffix, default=None):
        for key, value in meta.items():
            if key == suffix or key.endswith("." + suffix):
                return value
        return default

    layers = int(get("block_count", 0) or 0)
    interval = int(get("full_attention_interval", 0) or 0)
    types = []
    for i in range(layers):
        types.append("full_attention" if interval and (i + 1) % interval == 0 else "linear_attention")
    max_pos = int(get("context_length", DEFAULT_MAX_CTX) or DEFAULT_MAX_CTX)
    return {
        "kind": "gguf",
        "path": str(path),
        "dir": str(path.parent),
        "ok": True,
        "errors": [],
        "warnings": [],
        "layers": layers,
        "layer_types": types,
        "max_ctx": max_pos,
        "max_pos": max_pos,
        "experts": int(get("expert_count", 0) or 0),
        "experts_per_tok": int(get("expert_used_count", 0) or 0),
        "moe_i": int(get("expert_feed_forward_length", 0) or 0),
        "name": path.stem,
        "tied": get("output.weight") is None,
        "first_shard": None,
    }


def _read_hqm_i32(path, tensors, data_offset, name):
    if name not in tensors:
        return []
    raw = vk_llm.hqm_read_tensor(path, tensors, data_offset, name)
    if len(raw) % 4 != 0:
        return []
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def read_hqm_info(path):
    path = Path(path)
    kvs, tensors, data_offset = vk_llm.hqm_meta(path)
    types = _read_hqm_i32(path, tensors, data_offset, "config.layer_type")
    aq = _read_hqm_i32(path, tensors, data_offset, "config.layer_attn_quant")
    fq = _read_hqm_i32(path, tensors, data_offset, "config.layer_ffn_quant")
    max_ctx = int(kvs.get("hqm.max_ctx", DEFAULT_MAX_CTX) or DEFAULT_MAX_CTX)
    return {
        "kind": "hqm",
        "path": str(path),
        "dir": str(path.parent),
        "ok": True,
        "errors": [],
        "warnings": [],
        "layers": int(kvs.get("hqm.layer_count", 0) or 0),
        "layer_types": [_TYPE_BY_VALUE.get(v, "") for v in types],
        "attn_quants": [_QUANT_BY_VALUE.get(v, "fp16") for v in aq],
        "ffn_quants": [_QUANT_BY_VALUE.get(v, "fp16") for v in fq],
        "embed": _QUANT_BY_VALUE.get(int(kvs.get("hqm.embed_quant", 16) or 16), "fp16"),
        "lm_head": _QUANT_BY_VALUE.get(int(kvs.get("hqm.lm_head_quant", 16) or 16), "fp16"),
        "max_ctx": max_ctx,
        "max_pos": max_ctx,
        "prefill_chunk": int(kvs.get("hqm.prefill_chunk", DEFAULT_PREFILL) or DEFAULT_PREFILL),
        "experts": int(kvs.get("hqm.experts", 0) or 0),
        "name": str(kvs.get("general.name", path.stem)),
        "tied": bool(kvs.get("hqm.tied", 0)),
        "first_shard": None,
        "quant_baked": True,
    }


def _apply_existing_quant(info):
    quant_path = Path(info.get("dir", "")) / "quant_config.json"
    if not quant_path.is_file():
        return
    try:
        data = _read_json(quant_path)
    except Exception:
        return
    layers = data.get("layers", [])
    if len(layers) == int(info.get("layers", -1) or -1):
        info["attn_quants"] = [str(l.get("attn", "fp16")) for l in layers]
        info["ffn_quants"] = [str(l.get("ffn", "fp16")) for l in layers]
    embed = str(data.get("embed", "fp16"))
    lm_head = str(data.get("lm_head", "fp16"))
    info["embed"] = embed if embed in QUANT_VALUES else "fp16"
    info["lm_head"] = lm_head if lm_head in QUANT_VALUES else "fp16"
    info["prefill_chunk"] = int(data.get("prefill_chunk", DEFAULT_PREFILL) or DEFAULT_PREFILL)
    if "max_ctx" in data:
        info["max_ctx"] = int(data["max_ctx"])


def _probe(path):
    path = (path or "").strip()
    if not path:
        return {"kind": "missing", "path": "", "ok": False, "errors": ["no path selected"],
                "warnings": [], "layers": 0, "layer_types": []}
    kind = _model_kind(path)
    if kind in ("missing", "unknown"):
        return {"kind": kind, "path": path, "ok": False, "errors": ["unsupported model path"],
                "warnings": [], "layers": 0, "layer_types": []}
    try:
        if kind == "safetensors":
            p = Path(path)
            model_dir = p.parent if p.is_file() else p
            info = validate_safetensors(model_dir)
            info["selected"] = str(p)
        elif kind == "gguf":
            info = read_gguf_info(path)
        else:
            info = read_hqm_info(path)
    except Exception as exc:
        return {"kind": kind, "path": path, "ok": False, "errors": [str(exc)],
                "warnings": [], "layers": 0, "layer_types": []}
    if info.get("ok") and kind in ("safetensors", "gguf"):
        _apply_existing_quant(info)
    return info


def _layer_rows(info):
    count = int(info.get("layers", 0) or 0)
    types = info.get("layer_types") or []
    aq = info.get("attn_quants") or []
    fq = info.get("ffn_quants") or []
    rows = []
    for i in range(count):
        attn = aq[i] if i < len(aq) else "fp16"
        ffn = fq[i] if i < len(fq) else "fp16"
        rows.append([types[i] if i < len(types) else "", attn, ffn])
    return rows


def _cell_update(vals, index, field, value):
    data = [list(r) for r in (vals or [])]
    while len(data) <= index:
        data.append(["", "fp16", "fp16"])
    data[index][field] = value
    return data


def _collect_layers(vals, rows, count):
    source = vals if vals else rows
    layers = []
    for row in source or []:
        if len(row) < 3:
            continue
        attn = str(row[1]).strip().lower()
        ffn = str(row[2]).strip().lower()
        layers.append((attn if attn in QUANT_VALUES else "fp16", ffn if ffn in QUANT_VALUES else "fp16"))
    if count:
        if len(layers) > count:
            layers = layers[:count]
        while len(layers) < count:
            layers.append(("fp16", "fp16"))
    return layers


def on_model_change(path):
    info = _probe(path)
    _ENGINE["info"] = info
    rows = _layer_rows(info)
    locked = info.get("kind") == "hqm"
    tied = bool(info.get("tied", False))
    experts = int(info.get("experts", 0) or 0)
    max_ctx = int(info.get("max_ctx", DEFAULT_MAX_CTX) or DEFAULT_MAX_CTX)
    max_pos = int(info.get("max_pos", max_ctx) or max_ctx)
    embed = info.get("embed", "fp16")
    lm_head = info.get("lm_head", "fp16")
    prefill = int(info.get("prefill_chunk", DEFAULT_PREFILL) or DEFAULT_PREFILL)
    return (
        rows,
        rows,
        locked,
        gr.update(value=max_ctx, maximum=max_pos),
        gr.update(value=prefill),
        gr.update(value=embed, visible=not tied, interactive=not locked),
        gr.update(value=lm_head, visible=not tied, interactive=not locked),
        gr.update(value=embed, visible=tied, interactive=not locked),
        gr.update(value=experts, visible=experts > 0, interactive=experts > 0),
        gr.update(interactive=not locked),
        gr.update(interactive=not locked),
        gr.update(interactive=not locked),
    )


def apply_preset(rows, vals, preset_attn, preset_ffn):
    data = [list(r) for r in (rows or [])]
    for row in data:
        if preset_attn != "custom":
            row[1] = preset_attn
        if preset_ffn != "custom":
            row[2] = preset_ffn
    return data, data


def sync_max_new(limit, max_ctx):
    maximum = max(1, int(max_ctx))
    if limit:
        return gr.update(visible=True, maximum=maximum, value=maximum)
    return gr.update(visible=False)


def _close_engine():
    llm = _ENGINE.get("llm")
    if llm is not None:
        try:
            vk_llm.close(llm)
        except Exception:
            pass
    _ENGINE.update({"llm": None, "path": None, "kind": None, "info": {}})


def _resolve_dir(path):
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        p = p.parent
    return p


def _native_pick(kind):
    if kind == "dir":
        call = "filedialog.askdirectory(parent=r)"
    else:
        call = ("filedialog.askopenfilename(parent=r, filetypes=["
                "('Models', '*.gguf *.hqm *.safetensors'), ('All files', '*.*')])")
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk()\n"
        "r.withdraw()\n"
        "r.attributes('-topmost', True)\n"
        "try:\n"
        "    p = " + call + "\n"
        "except Exception:\n"
        "    p = ''\n"
        "finally:\n"
        "    r.destroy()\n"
        "print(p or '')\n"
    )
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, timeout=600, creationflags=flags)
    except Exception:
        return ""
    lines = out.stdout.strip().splitlines()
    return lines[-1] if lines else ""


def pick_model_file():
    return gr.update(value=_native_pick("file"))


def pick_export_dir():
    return gr.update(value=_native_pick("dir"))


def load_model(path, vals, rows, max_ctx, prefill_chunk, embed_q, lm_head_q, embed_lm_q,
               experts_vram, prune, do_export, export_dir):
    with _ENGINE["lock"]:
        info = _probe(path)
        if not info.get("ok"):
            return "Validation failed: " + "; ".join(info.get("errors", ["unknown error"]))
        kind = info["kind"]
        weight_dir = path
        if kind == "safetensors":
            weight_dir = str(_resolve_dir(path) or path)
        quant_config = None
        if kind != "hqm":
            if not vals and not rows:
                rows = _layer_rows(info)
            layers = _collect_layers(vals, rows, int(info.get("layers", 0) or 0))
            tied = bool(info.get("tied", False))
            embed = embed_lm_q if tied else embed_q
            lm_head = embed_lm_q if tied else lm_head_q
            cfg = {
                "name": info.get("name", "model"),
                "max_ctx": int(max_ctx),
                "prefill_chunk": int(prefill_chunk),
                "embed": embed,
                "lm_head": lm_head,
                "layers": [{"attn": a, "ffn": f} for a, f in layers],
            }
            is_moe = int(info.get("experts", 0) or 0) > 0
            if is_moe:
                cfg["experts_vram"] = int(experts_vram)
            QUANT_TMP.parent.mkdir(parents=True, exist_ok=True)
            QUANT_TMP.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            quant_config = QUANT_TMP
        _close_engine()
        out_dir = _resolve_dir(export_dir) or DEFAULT_EXPORT_DIR
        try:
            llm = vk_llm.start_llm(
                weight_dir,
                max_ctx=int(max_ctx),
                max_new_tokens=4096,
                prune_vocab=bool(prune) and kind != "hqm",
                experts_vram=int(experts_vram) if int(info.get("experts", 0) or 0) > 0 else 0,
                export=bool(do_export),
                export_dir=str(out_dir) if do_export else None,
                quant_config=quant_config,
            )
        except Exception as exc:
            return "Load failed: %s" % exc
        _ENGINE.update({"llm": llm, "path": path, "kind": kind, "info": info})
        return "Loaded %s (%s, %d layers)" % (path, kind, int(info.get("layers", 0) or 0))


def unload_model():
    with _ENGINE["lock"]:
        _close_engine()
    return "Unloaded"


def _set_sampling(llm, temperature, top_k, top_p, min_p, rep_penalty, penalty_len, presence_penalty, seed, greedy):
    vk_llm.set_sampling(
        llm,
        is_sampling=not greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        rep_penalty=rep_penalty,
        penalty_len=penalty_len,
        presence_penalty=presence_penalty,
        seed=seed,
    )


def _visible(text, hide):
    if not hide:
        return text
    text = _THINK_RE.sub("", text)
    cut = text.find("<think>")
    if cut >= 0:
        text = text[:cut]
    return text


def respond(message, history, system_prompt, thinking, hide_thinking, max_new, limit_max_new, max_ctx,
            temperature, top_k, top_p, min_p, rep_penalty, penalty_len, presence_penalty, seed, sampling):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        yield history, "", ""
        return
    llm = _ENGINE.get("llm")
    if llm is None:
        yield history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "No model loaded."}], "", ""
        return
    llm.max_new_tokens = int(max_new) if limit_max_new else int(max_ctx)
    _set_sampling(llm, temperature, top_k, top_p, min_p, rep_penalty, penalty_len, presence_penalty, seed, not sampling)
    messages = history + [{"role": "user", "content": message}]
    try:
        ids = vk_llm.apply_chat_template(llm, messages, system=system_prompt or "", enable_thinking=bool(thinking))
    except Exception as exc:
        yield history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "Tokenize failed: %s" % exc}], "", ""
        return
    partial = ""
    count = 0
    start = time.perf_counter()
    try:
        for delta, n in vk_llm.generate_stream_tokens(llm, ids):
            count = n
            partial += delta
            elapsed = time.perf_counter() - start
            rate = "%d tokens, %.1f tok/s" % (count, count / elapsed) if elapsed > 0 else ""
            yield history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": _visible(partial, hide_thinking)}], "", rate
    except Exception as exc:
        yield history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "Generation failed: %s" % exc}], "", ""
        return
    elapsed = time.perf_counter() - start
    rate = "%d tokens, %.1f tok/s" % (count, count / elapsed) if elapsed > 0 else ""
    yield history + [{"role": "user", "content": message},
                     {"role": "assistant", "content": _visible(partial, hide_thinking)}], "", rate


def clear_chat():
    return [], "", ""


def build_ui():
    DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with gr.Blocks(title="VK Compute") as demo:
        gr.Markdown("# VK Compute")
        layer_rows = gr.State([])
        layer_vals = gr.State([])
        locked = gr.State(False)
        with gr.Tabs():
            with gr.Tab("Model"):
                with gr.Row():
                    model_path = gr.Textbox(label="Model path", scale=4)
                    pick_file_btn = gr.Button("Choose file", scale=1)
                with gr.Accordion("Quantization", open=False):
                    with gr.Row():
                        preset_attn = gr.Dropdown(label="Set all attn", choices=PRESET_CHOICES, value="custom")
                        preset_ffn = gr.Dropdown(label="Set all ffn", choices=PRESET_CHOICES, value="custom")
                        preset_btn = gr.Button("Apply preset")
                    with gr.Row():
                        gr.Markdown("**layer**")
                        gr.Markdown("**type**")
                        gr.Markdown("**attn**")
                        gr.Markdown("**ffn**")

                    @gr.render(inputs=[layer_rows, locked])
                    def render_layers(rows, is_locked):
                        for i, row in enumerate(rows):
                            with gr.Row():
                                gr.Textbox(value=str(i), show_label=False, scale=1, interactive=False)
                                gr.Textbox(value=row[0], show_label=False, scale=2, interactive=False)
                                attn = gr.Dropdown(choices=QUANT_CHOICES, value=row[1], show_label=False,
                                                   scale=2, interactive=not is_locked,
                                                   elem_classes=["locked"] if is_locked else [])
                                ffn = gr.Dropdown(choices=QUANT_CHOICES, value=row[2], show_label=False,
                                                  scale=2, interactive=not is_locked,
                                                  elem_classes=["locked"] if is_locked else [])
                                attn.change(lambda v, s, i=i: _cell_update(s, i, 1, v),
                                            [attn, layer_vals], [layer_vals])
                                ffn.change(lambda v, s, i=i: _cell_update(s, i, 2, v),
                                           [ffn, layer_vals], [layer_vals])

                with gr.Row():
                    max_ctx = gr.Number(label="Max context", value=DEFAULT_MAX_CTX, precision=0)
                    prefill_chunk = gr.Number(label="Prefill chunk", value=DEFAULT_PREFILL, precision=0)
                with gr.Row():
                    embed_q = gr.Dropdown(label="Embed quant", choices=QUANT_CHOICES, value="fp16")
                    lm_head_q = gr.Dropdown(label="LM head quant", choices=QUANT_CHOICES, value="fp16")
                    embed_lm_q = gr.Dropdown(label="Embed/LM head quant", choices=QUANT_CHOICES,
                                             value="fp16", visible=False)
                    experts_vram = gr.Number(label="Experts in VRAM", value=0, precision=0, visible=False)
                with gr.Row():
                    prune = gr.Checkbox(label="Prune vocab", value=False)
                    do_export = gr.Checkbox(label="Export HQM", value=True)
                with gr.Row():
                    export_dir = gr.Textbox(label="Export dir", value=str(DEFAULT_EXPORT_DIR), scale=4)
                    pick_export_btn = gr.Button("Choose folder", scale=1)
                with gr.Row():
                    load_btn = gr.Button("Load / Export HQM", variant="primary")
                    unload_btn = gr.Button("Unload")
                load_status = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("Sampling"):
                with gr.Row():
                    sampling = gr.Checkbox(label="Sampling", value=False)
                    thinking = gr.Checkbox(label="Thinking", value=True)
                    hide_thinking = gr.Checkbox(label="Hide thinking in chat", value=False)
                with gr.Column(visible=False) as sampling_box:
                    with gr.Row():
                        temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.01, label="Temperature")
                        top_k = gr.Slider(0, 200, value=20, step=1, label="Top-k")
                        top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                        min_p = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Min-p")
                    with gr.Row():
                        rep_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition penalty")
                        penalty_len = gr.Slider(0, 1024, value=64, step=1, label="Penalty length")
                        presence_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.01, label="Presence penalty")
                        seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
                limit_max_new = gr.Checkbox(label="Max new tokens", value=False)
                max_new = gr.Slider(1, DEFAULT_MAX_CTX, value=1024, step=1,
                                    label="Max new tokens", visible=False)
                system_prompt = gr.Textbox(label="System prompt", value="You are a helpful assistant.")

            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Chat", height=520)
                with gr.Row():
                    msg = gr.Textbox(label="Message", scale=4, lines=1)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("Clear session")
                    chat_status = gr.Textbox(label="Status", interactive=False, scale=3)

        pick_file_btn.click(pick_model_file, outputs=[model_path]).then(
            on_model_change,
            inputs=[model_path],
            outputs=[layer_rows, layer_vals, locked, max_ctx, prefill_chunk, embed_q, lm_head_q,
                     embed_lm_q, experts_vram, preset_attn, preset_ffn, preset_btn],
        )
        model_path.submit(
            on_model_change,
            inputs=[model_path],
            outputs=[layer_rows, layer_vals, locked, max_ctx, prefill_chunk, embed_q, lm_head_q,
                     embed_lm_q, experts_vram, preset_attn, preset_ffn, preset_btn],
        )
        pick_export_btn.click(pick_export_dir, outputs=[export_dir])
        preset_btn.click(apply_preset, inputs=[layer_rows, layer_vals, preset_attn, preset_ffn],
                         outputs=[layer_rows, layer_vals])
        load_btn.click(
            load_model,
            inputs=[model_path, layer_vals, layer_rows, max_ctx, prefill_chunk, embed_q, lm_head_q,
                    embed_lm_q, experts_vram, prune, do_export, export_dir],
            outputs=[load_status],
        )
        unload_btn.click(unload_model, outputs=[load_status])
        sampling.change(lambda s: gr.update(visible=s), [sampling], [sampling_box])
        limit_max_new.change(sync_max_new, [limit_max_new, max_ctx], [max_new])
        max_ctx.change(sync_max_new, [limit_max_new, max_ctx], [max_new])

        chat_inputs = [msg, chatbot, system_prompt, thinking, hide_thinking, max_new, limit_max_new, max_ctx,
                       temperature, top_k, top_p, min_p, rep_penalty, penalty_len,
                       presence_penalty, seed, sampling]
        msg.submit(respond, inputs=chat_inputs, outputs=[chatbot, msg, chat_status])
        send_btn.click(respond, inputs=chat_inputs, outputs=[chatbot, msg, chat_status])
        clear_btn.click(clear_chat, outputs=[chatbot, msg, chat_status])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share, css=_CSS)


if __name__ == "__main__":
    main()
