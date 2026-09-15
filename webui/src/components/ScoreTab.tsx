import { useRef, useState } from "react";
import { scoreStream, stopScore, type ScoreDone, type ScoreProgress } from "../api";
interface Props {
  loaded: boolean;
}

export function ScoreTab({ loaded }: Props) {
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");
  const [prefill, setPrefill] = useState(4096);
  const [decode, setDecode] = useState(4096);
  const [sizePct, setSizePct] = useState(10);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ScoreProgress | null>(null);
  const [result, setResult] = useState<ScoreDone | null>(null);
  const [status, setStatus] = useState("");
  const stoppedRef = useRef(false);

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setFileName(file.name);
    setText(await file.text());
    setResult(null);
    setProgress(null);
    setStatus("");
  };

  const run = async () => {
    if (!text.trim() || running) return;
    setRunning(true);
    setResult(null);
    setProgress(null);
    setStatus("scoring...");
    stoppedRef.current = false;
    try {
      const done = await scoreStream(
        { text, prefill, decode, sizePct: sizePct / 100 },
        (p) => setProgress(p),
      );
      setResult(done);
      setStatus(stoppedRef.current ? "stopped (partial result)" : "done");
    } catch (e) {
      setStatus(String(e));
    } finally {
      setRunning(false);
    }
  };

  const stop = () => {
    stoppedRef.current = true;
    setStatus("stopping...");
    stopScore();
  };
  const pct = progress && progress.total > 0 ? (progress.done / progress.total) * 100 : 0;

  return (
    <div className="score">
      <div className="row">
        <label className="field grow">
          Text file
          <input type="file" accept=".txt,text/plain"
            onChange={(e) => pick(e.target.files?.[0])} />
        </label>
        <span className="muted">{fileName}</span>
      </div>

      <div className="grid">
        <label className="field">
          Prefill tokens
          <input type="number" min={1} value={prefill}
            onChange={(e) => setPrefill(Math.max(1, Number(e.target.value)))} />
        </label>
        <label className="field">
          Decode tokens
          <input type="number" min={1} value={decode}
            onChange={(e) => setDecode(Math.max(1, Number(e.target.value)))} />
        </label>
        <label className="field">
          Test size (% of file)
          <input type="number" min={0.1} max={100} step={0.1} value={sizePct}
            onChange={(e) => setSizePct(Number(e.target.value))} />
        </label>
      </div>

      <div className="row">
        {running ? (
          <button className="stop" onClick={stop}>Stop</button>
        ) : (
          <button onClick={run} disabled={!loaded || !text.trim()}>Run</button>
        )}
        {!loaded && <span className="muted">load a model first</span>}
      </div>

      {running && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${pct}%` }} />
          <span className="progress-text">
            chunk {progress?.done ?? 0}/{progress?.total ?? 0}
            {progress ? ` · ppl ${progress.ppl.toFixed(3)} · ${progress.count} tokens` : ""}
          </span>
        </div>
      )}

      {result && (
        <div className="note ok">
          <div>Perplexity: <b>{result.ppl.toFixed(4)}</b></div>
          <div>Mean NLL: {(result.loss / result.count).toFixed(4)}</div>
          <div>Scored tokens: {result.count}</div>
          <div>File tokens: {result.tokens} · {((result.elapsedMs / 1000) / (result.count / 1000)).toFixed(2)} ms/token</div>
        </div>
      )}

      <div className="status">{status}</div>
    </div>
  );
}
