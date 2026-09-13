import type { LayerRow, Quant } from "../../server/types";

const QUANTS: Quant[] = ["fp16", "int8", "int4"];
const PRESETS: (Quant | "custom")[] = ["custom", "fp16", "int8", "int4"];

interface Props {
  layers: LayerRow[];
  locked: boolean;
  onChange: (layers: LayerRow[]) => void;
}

export function QuantEditor({ layers, locked, onChange }: Props) {
  const setAll = (field: "attn" | "ffn", value: Quant) => {
    onChange(layers.map((row) => ({ ...row, [field]: value })));
  };

  const setCell = (index: number, field: "attn" | "ffn", value: Quant) => {
    onChange(layers.map((row, i) => (i === index ? { ...row, [field]: value } : row)));
  };

  return (
    <div className={locked ? "quant locked" : "quant"}>
      <div className="quant-presets">
        <label>
          Set all attn
          <select value="custom" disabled={locked}
            onChange={(e) => e.target.value !== "custom" && setAll("attn", e.target.value as Quant)}>
            {PRESETS.map((q) => <option key={q} value={q}>{q === "custom" ? "custom" : q.toUpperCase()}</option>)}
          </select>
        </label>
        <label>
          Set all ffn
          <select value="custom" disabled={locked}
            onChange={(e) => e.target.value !== "custom" && setAll("ffn", e.target.value as Quant)}>
            {PRESETS.map((q) => <option key={q} value={q}>{q === "custom" ? "custom" : q.toUpperCase()}</option>)}
          </select>
        </label>
      </div>
      <div className="quant-table">
        <div className="quant-head">
          <span>layer</span><span>type</span><span>attn</span><span>ffn</span>
        </div>
        {layers.map((row, i) => (
          <div className="quant-row" key={i}>
            <span className="quant-index">{i}</span>
            <span className="quant-type">{row.type}</span>
            <select value={row.attn} disabled={locked}
              onChange={(e) => setCell(i, "attn", e.target.value as Quant)}>
              {QUANTS.map((q) => <option key={q} value={q}>{q.toUpperCase()}</option>)}
            </select>
            <select value={row.ffn} disabled={locked}
              onChange={(e) => setCell(i, "ffn", e.target.value as Quant)}>
              {QUANTS.map((q) => <option key={q} value={q}>{q.toUpperCase()}</option>)}
            </select>
          </div>
        ))}
        {!layers.length && <div className="quant-empty">No model selected</div>}
      </div>
    </div>
  );
}
