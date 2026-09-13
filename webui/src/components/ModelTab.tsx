import { useState } from "react";
import type { LayerRow, ProbeInfo, Quant } from "../../server/types";
import type { ModelEntry } from "../api";
import { QuantEditor } from "./QuantEditor";

export interface ModelForm {
  path: string;
  maxCtx: number;
  prefillChunk: number;
  embed: Quant;
  lmHead: Quant;
  embedLm: Quant;
  expertsVram: number;
  prune: boolean;
  exportModel: boolean;
  exportDir: string;
}

interface Props {
  models: ModelEntry[];
  info: ProbeInfo | null;
  layers: LayerRow[];
  form: ModelForm;
  status: string;
  loading: boolean;
  loaded: boolean;
  onRefresh: () => void;
  onSelect: (path: string) => void;
  onProbe: (path: string) => void;
  onForm: (patch: Partial<ModelForm>) => void;
  onLayers: (layers: LayerRow[]) => void;
  onLoad: () => void;
  onUnload: () => void;
}

export function ModelTab(props: Props) {
  const { models, info, layers, form, status, loading, loaded } = props;
  const [quantOpen, setQuantOpen] = useState(false);
  const locked = info?.kind === "hqm";
  const tied = Boolean(info?.tied);
  const isMoe = (info?.experts ?? 0) > 0;

  return (
    <div className="model">
      <div className="row">
        <select value={form.path} onChange={(e) => props.onSelect(e.target.value)}>
          <option value="">select model...</option>
          {models.map((m) => (
            <option key={m.path} value={m.path}>{m.label}</option>
          ))}
        </select>
        <button className="secondary" onClick={props.onRefresh}>Refresh</button>
      </div>
      <div className="row">
        <input className="grow" type="text" placeholder="model path" value={form.path}
          onChange={(e) => props.onForm({ path: e.target.value })}
          onBlur={(e) => props.onProbe(e.target.value)} />
        <button className="secondary" onClick={() => props.onProbe(form.path)}>Validate</button>
      </div>

      {info && (
        <div className={info.ok ? "note ok" : "note error"}>
          <b>{info.kind}</b>
          {info.ok ? ` — ${info.layers} layers` : ""}
          {info.errors.length > 0 && <div>{info.errors.join("; ")}</div>}
          {info.warnings.length > 0 && <div className="muted">{info.warnings.join("; ")}</div>}
        </div>
      )}

      <details className="accordion" open={quantOpen} onToggle={(e) => setQuantOpen((e.target as HTMLDetailsElement).open)}>
        <summary>Quantization</summary>
        <QuantEditor layers={layers} locked={locked} onChange={props.onLayers} />
      </details>

      <div className="grid">
        <label className="field">
          Max context
          <input type="number" value={form.maxCtx} min={1024} max={info?.maxPos ?? 32768}
            onChange={(e) => props.onForm({ maxCtx: Number(e.target.value) })} />
        </label>
        <label className="field">
          Prefill chunk
          <input type="number" value={form.prefillChunk} min={16}
            onChange={(e) => props.onForm({ prefillChunk: Number(e.target.value) })} />
        </label>
        {tied ? (
          <label className="field">
            Embed/LM head quant
            <select value={form.embedLm} disabled={locked}
              onChange={(e) => props.onForm({ embedLm: e.target.value as Quant })}>
              {["fp16", "int8", "int4"].map((q) => <option key={q} value={q}>{q.toUpperCase()}</option>)}
            </select>
          </label>
        ) : (
          <>
            <label className="field">
              Embed quant
              <select value={form.embed} disabled={locked}
                onChange={(e) => props.onForm({ embed: e.target.value as Quant })}>
                {["fp16", "int8", "int4"].map((q) => <option key={q} value={q}>{q.toUpperCase()}</option>)}
              </select>
            </label>
            <label className="field">
              LM head quant
              <select value={form.lmHead} disabled={locked}
                onChange={(e) => props.onForm({ lmHead: e.target.value as Quant })}>
                {["fp16", "int8", "int4"].map((q) => <option key={q} value={q}>{q.toUpperCase()}</option>)}
              </select>
            </label>
          </>
        )}
        {isMoe && (
          <label className="field">
            Experts in VRAM
            <input type="number" value={form.expertsVram} min={1} max={info?.experts ?? 0}
              onChange={(e) => props.onForm({ expertsVram: Number(e.target.value) })} />
          </label>
        )}
      </div>

      <div className="row">
        <label className="check">
          <input type="checkbox" checked={form.prune} disabled={info?.kind === "hqm"}
            onChange={(e) => props.onForm({ prune: e.target.checked })} />
          Prune vocab
        </label>
        <label className="check">
          <input type="checkbox" checked={form.exportModel}
            onChange={(e) => props.onForm({ exportModel: e.target.checked })} />
          Export HQM
        </label>
        <input className="grow" type="text" placeholder="export dir" value={form.exportDir}
          onChange={(e) => props.onForm({ exportDir: e.target.value })} />
      </div>

      <div className="row">
        <button onClick={props.onLoad} disabled={loading || !form.path}>
          {loading ? "Loading..." : "Load / Export HQM"}
        </button>
        <button className="secondary" onClick={props.onUnload} disabled={!loaded}>Unload</button>
        <span className={loaded ? "note ok" : "muted"}>{loaded ? "loaded" : "not loaded"}</span>
      </div>
      <div className="status">{status}</div>
    </div>
  );
}
