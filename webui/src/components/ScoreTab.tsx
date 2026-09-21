import { useEffect, useRef, useState } from "react";
import {
  scoreStream,
  stopScore,
  type ScoreDone,
  type ScoreProgress,
  type ScoreStart,
} from "../api";
import { NumberInput } from "./NumberInput";

interface Props {
  loaded: boolean;
}

interface Settings {
  prefill: number;
  decode: number;
  sizePct: number;
  fileName: string;
}

interface ChunkStat {
  chunk: number;
  ppl: number;
  count: number;
}

const STORAGE_KEY = "pumice-eval-settings";

const DEFAULT_SETTINGS: Settings = {
  prefill: 4096,
  decode: 4096,
  sizePct: 10,
  fileName: "",
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {}
  return DEFAULT_SETTINGS;
}

function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--:--";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function safeFileName(name: string): string {
  return name.replace(/[\\/:*?"<>|]/g, "_");
}

export function ScoreTab({ loaded }: Props) {
  const [initial] = useState(loadSettings);
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState(initial.fileName);
  const [prefill, setPrefill] = useState(initial.prefill);
  const [decode, setDecode] = useState(initial.decode);
  const [sizePct, setSizePct] = useState(initial.sizePct);
  const [running, setRunning] = useState(false);
  const [plan, setPlan] = useState<ScoreStart | null>(null);
  const [progress, setProgress] = useState<ScoreProgress | null>(null);
  const [chunks, setChunks] = useState<ChunkStat[]>([]);
  const [result, setResult] = useState<ScoreDone | null>(null);
  const [status, setStatus] = useState("");
  const startedAt = useRef(0);
  const stoppedRef = useRef(false);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ prefill, decode, sizePct, fileName }));
  }, [prefill, decode, sizePct, fileName]);

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setFileName(file.name);
    setText(await file.text());
    setResult(null);
    setProgress(null);
    setChunks([]);
    setPlan(null);
    setStatus("");
  };

  const run = async () => {
    if (!text.trim() || running) return;
    setRunning(true);
    setResult(null);
    setProgress(null);
    setChunks([]);
    setPlan(null);
    setStatus("tokenizing...");
    stoppedRef.current = false;
    startedAt.current = Date.now();
    try {
      const done = await scoreStream(
        { text, prefill, decode, sizePct: sizePct / 100 },
        {
          onStart: (s) => {
            setPlan(s);
            setStatus("scoring...");
          },
          onProgress: (p) => {
            setProgress(p);
            setChunks((prev) =>
              p.done > prev.length ? [...prev, { chunk: p.done, ppl: p.ppl, count: p.count }] : prev,
            );
          },
        },
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

  const download = () => {
    if (!result || !plan || result.count <= 0) return;
    const seconds = result.elapsedMs / 1000;
    const lines = [
      `model: ${plan.model}`,
      `file: ${fileName || "(none)"}`,
      `file tokens: ${plan.fileTokens}`,
      `scored tokens: ${result.count}`,
      `prefill: ${plan.prefill}`,
      `decode: ${plan.decode}`,
      `chunks: ${result.chunks}`,
      `perplexity: ${result.ppl.toFixed(4)}`,
      `mean nll: ${(result.loss / result.count).toFixed(4)}`,
      `elapsed: ${seconds.toFixed(1)} s`,
      `tokens/s: ${(result.count / seconds).toFixed(2)}`,
      "",
      "chunk\tperplexity",
      ...chunks.map((c) => `${c.chunk}\t${c.ppl.toFixed(4)}`),
      "",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safeFileName(plan.model)}-ppl.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const scored = progress?.count ?? 0;
  const elapsed = running && startedAt.current ? (Date.now() - startedAt.current) / 1000 : 0;
  const rate = elapsed > 0 ? scored / elapsed : 0;
  const remaining = plan && rate > 0 ? (plan.scoredTokens - scored) / rate : NaN;
  const overallPct = plan && plan.scoredTokens > 0 ? (scored / plan.scoredTokens) * 100 : 0;
  const chunkPct = progress && progress.chunkTotal > 0 ? (progress.chunkTokens / progress.chunkTotal) * 100 : 0;
  const totalChunks = plan?.chunks ?? progress?.total ?? 0;
  const currentChunk = Math.min((progress?.done ?? 0) + 1, Math.max(totalChunks, 1));
  const nll = progress && progress.count > 0 ? progress.loss / progress.count : 0;

  return (
    <div className="score">
      <div className="row">
        <label className="field grow">
          Text file
          <input type="file" accept=".txt,text/plain"
            onChange={(e) => pick(e.target.files?.[0])} />
        </label>
        <span className="muted">{fileName}{text ? "" : " (re-select to load)"}</span>
      </div>

      <div className="grid">
        <label className="field">
          Prefill tokens
          <NumberInput value={prefill} min={1} integer onChange={setPrefill} />
        </label>
        <label className="field">
          Decode tokens
          <NumberInput value={decode} min={1} integer onChange={setDecode} />
        </label>
        <label className="field">
          Test size (% of file)
          <NumberInput value={sizePct} min={0.1} max={100} onChange={setSizePct} />
        </label>
      </div>

      <div className="row">
        {running ? (
          <button className="stop" onClick={stop}>Stop</button>
        ) : (
          <button onClick={run} disabled={!loaded || !text.trim()}>Run</button>
        )}
        {result && (
          <button className="secondary" onClick={download} disabled={result.count <= 0}>Download</button>
        )}
        {!loaded && <span className="muted">load a model first</span>}
        {plan && (
          <span className="muted">
            {plan.fileTokens} file tokens · scoring {plan.scoredTokens} · {plan.chunks} chunks
          </span>
        )}
      </div>

      {running && (
        <div className="progress-group">
          <div className="progress-label">
            <span>Overall · chunk {progress?.done ?? 0}/{totalChunks}</span>
            <span>
              {scored} tokens · ppl {progress ? progress.ppl.toFixed(3) : "-"} · nll {progress ? nll.toFixed(3) : "-"}
              {" · "}{rate.toFixed(1)} tok/s · ETA {fmtDuration(remaining)}
            </span>
          </div>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.min(100, overallPct)}%` }} />
          </div>

          <div className="progress-label">
            <span>Chunk {currentChunk}/{totalChunks}</span>
            <span>
              {progress
                ? progress.chunkTokens === 0
                  ? `prefill ${progress.chunkTotal} tokens`
                  : `${progress.chunkTokens}/${progress.chunkTotal} tokens`
                : "waiting"}
            </span>
          </div>
          <div className="progress thin">
            <div className="progress-bar" style={{ width: `${Math.min(100, chunkPct)}%` }} />
          </div>
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
