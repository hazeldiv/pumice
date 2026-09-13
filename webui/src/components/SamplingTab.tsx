export interface SamplingState {
  enabled: boolean;
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  repPenalty: number;
  penaltyLength: number;
  presencePenalty: number;
  seed: number;
  limitMaxNew: boolean;
  maxNew: number;
  thinking: boolean;
  hideThinking: boolean;
  system: string;
  maxCtx: number;
}

interface Props {
  state: SamplingState;
  onChange: (patch: Partial<SamplingState>) => void;
}

function Num({ label, value, min, max, step, onChange, disabled }: {
  label: string; value: number; min?: number; max?: number; step?: number;
  onChange: (v: number) => void; disabled?: boolean;
}) {
  return (
    <label className="field">
      {label}
      <input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label className="field slider">
      <span>{label} <b>{value}</b></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function SamplingTab({ state, onChange }: Props) {
  return (
    <div className="sampling">
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={state.enabled}
            onChange={(e) => onChange({ enabled: e.target.checked })} />
          Sampling
        </label>
        <label className="check">
          <input type="checkbox" checked={state.thinking}
            onChange={(e) => onChange({ thinking: e.target.checked })} />
          Thinking
        </label>
        <label className="check">
          <input type="checkbox" checked={state.hideThinking}
            onChange={(e) => onChange({ hideThinking: e.target.checked })} />
          Hide thinking in chat
        </label>
      </div>

      {state.enabled && (
        <div className="panel">
          <Slider label="Temperature" value={state.temperature} min={0} max={2} step={0.01}
            onChange={(v) => onChange({ temperature: v })} />
          <Slider label="Top-k" value={state.topK} min={0} max={200} step={1}
            onChange={(v) => onChange({ topK: v })} />
          <Slider label="Top-p" value={state.topP} min={0} max={1} step={0.01}
            onChange={(v) => onChange({ topP: v })} />
          <Slider label="Min-p" value={state.minP} min={0} max={1} step={0.01}
            onChange={(v) => onChange({ minP: v })} />
          <Slider label="Repetition penalty" value={state.repPenalty} min={1} max={2} step={0.01}
            onChange={(v) => onChange({ repPenalty: v })} />
          <Slider label="Penalty length" value={state.penaltyLength} min={0} max={1024} step={1}
            onChange={(v) => onChange({ penaltyLength: v })} />
          <Slider label="Presence penalty" value={state.presencePenalty} min={0} max={2} step={0.01}
            onChange={(v) => onChange({ presencePenalty: v })} />
          <Num label="Seed (0 = random)" value={state.seed} min={0}
            onChange={(v) => onChange({ seed: v })} />
        </div>
      )}

      <label className="check">
        <input type="checkbox" checked={state.limitMaxNew}
          onChange={(e) => onChange({ limitMaxNew: e.target.checked })} />
        Max new tokens
      </label>
      {state.limitMaxNew && (
        <Slider label="Max new tokens" value={state.maxNew} min={1} max={state.maxCtx} step={1}
          onChange={(v) => onChange({ maxNew: v })} />
      )}

      <label className="field">
        System prompt
        <textarea rows={2} value={state.system}
          onChange={(e) => onChange({ system: e.target.value })} />
      </label>
    </div>
  );
}
