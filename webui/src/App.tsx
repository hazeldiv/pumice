import { useEffect, useState } from "react";
import { chatStream, listModels, loadModel, probeModel, unloadModel, type ModelEntry } from "./api";
import type { ChatMessage, LayerRow, ProbeInfo, Quant } from "../server/types";
import { ModelTab, type ModelForm } from "./components/ModelTab";
import { SamplingTab, type SamplingState } from "./components/SamplingTab";
import { ChatTab } from "./components/ChatTab";

type Tab = "model" | "sampling" | "chat";

const DEFAULT_FORM: ModelForm = {
  path: "",
  maxCtx: 32768,
  prefillChunk: 512,
  embed: "fp16",
  lmHead: "fp16",
  embedLm: "fp16",
  expertsVram: 0,
  prune: false,
  exportModel: true,
  exportDir: "exported",
};

const DEFAULT_SAMPLING: SamplingState = {
  enabled: false,
  temperature: 0.6,
  topK: 20,
  topP: 0.95,
  minP: 0,
  repPenalty: 1.05,
  penaltyLength: 64,
  presencePenalty: 0,
  seed: 0,
  limitMaxNew: false,
  maxNew: 1024,
  thinking: true,
  hideThinking: false,
  system: "You are a helpful assistant.",
  maxCtx: 32768,
};

function visible(text: string, hide: boolean): string {
  if (!hide) return text;
  const closed = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const open = closed.indexOf("<think>");
  return open >= 0 ? closed.slice(0, open) : closed;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("model");
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [info, setInfo] = useState<ProbeInfo | null>(null);
  const [layers, setLayers] = useState<LayerRow[]>([]);
  const [form, setForm] = useState<ModelForm>(DEFAULT_FORM);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [sampling, setSampling] = useState<SamplingState>(DEFAULT_SAMPLING);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [chatStatus, setChatStatus] = useState("");

  const refreshModels = async () => {
    try {
      setModels(await listModels());
    } catch (e) {
      setStatus(String(e));
    }
  };

  useEffect(() => {
    refreshModels();
  }, []);

  const applyInfo = (p: ProbeInfo) => {
    setInfo(p);
    setLayers(
      Array.from({ length: p.layers }, (_, i) => ({
        type: p.layerTypes[i] ?? "",
        attn: (p.attnQuants[i] ?? "fp16") as Quant,
        ffn: (p.ffnQuants[i] ?? "fp16") as Quant,
      })),
    );
    setForm((f) => ({
      ...f,
      path: p.path,
      maxCtx: p.maxCtx,
      prefillChunk: p.prefillChunk,
      embed: p.embed,
      lmHead: p.lmHead,
      embedLm: p.embed,
      expertsVram: p.experts > 0 ? Math.min(f.expertsVram || p.experts, p.experts) : 0,
    }));
    setSampling((s) => ({ ...s, maxCtx: p.maxCtx, maxNew: Math.min(s.maxNew, p.maxCtx) }));
  };

  const probe = async (path: string) => {
    if (!path) return;
    setStatus("validating...");
    try {
      const p = await probeModel(path);
      applyInfo(p);
      setStatus(p.ok ? "valid" : "validation failed");
    } catch (e) {
      setStatus(String(e));
    }
  };

  const selectModel = async (path: string) => {
    setForm((f) => ({ ...f, path }));
    await probe(path);
  };

  const updateForm = (patch: Partial<ModelForm>) => setForm((f) => ({ ...f, ...patch }));

  const load = async () => {
    setLoading(true);
    setStatus("loading...");
    try {
      const result = await loadModel({
        path: form.path,
        maxCtx: form.maxCtx,
        prefillChunk: form.prefillChunk,
        embed: info?.tied ? form.embedLm : form.embed,
        lmHead: info?.tied ? form.embedLm : form.lmHead,
        expertsVram: form.expertsVram,
        prune: form.prune,
        exportModel: form.exportModel,
        exportDir: form.exportDir,
        layers: layers.map((l) => ({ attn: l.attn, ffn: l.ffn })),
      });
      setLoaded(true);
      setStatus(`loaded ${result.info.name} (${result.info.kind})`);
    } catch (e) {
      setStatus(String(e));
    } finally {
      setLoading(false);
    }
  };

  const unload = async () => {
    await unloadModel();
    setLoaded(false);
    setStatus("unloaded");
  };

  const send = async (text: string) => {
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setStreaming(true);
    setChatStatus("generating...");
    let acc = "";
    try {
      const done = await chatStream(
        {
          messages: history,
          system: sampling.system,
          thinking: sampling.thinking,
          sampling: {
            temperature: sampling.enabled ? sampling.temperature : 0,
            repPenalty: sampling.enabled ? sampling.repPenalty : 1,
            penaltyLength: sampling.enabled ? sampling.penaltyLength : 0,
            topK: sampling.enabled ? sampling.topK : 0,
            topP: sampling.enabled ? sampling.topP : 1,
            minP: sampling.enabled ? sampling.minP : 0,
            presencePenalty: sampling.enabled ? sampling.presencePenalty : 0,
            seed: sampling.seed,
          },
          maxNew: sampling.limitMaxNew ? sampling.maxNew : sampling.maxCtx,
        },
        (delta) => {
          acc += delta;
          setMessages([...history, { role: "assistant", content: visible(acc, sampling.hideThinking) }]);
        },
      );
      const rate = done.elapsedMs > 0 ? (done.tokens / (done.elapsedMs / 1000)).toFixed(1) : "0";
      setChatStatus(`${done.tokens} tokens, ${rate} tok/s`);
    } catch (e) {
      setChatStatus(String(e));
      setMessages([...history, { role: "assistant", content: `Error: ${e}` }]);
    } finally {
      setStreaming(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setChatStatus("");
  };

  return (
    <div className="app">
      <header>
        <h1>VK Compute</h1>
        <nav>
          <button className={tab === "model" ? "active" : ""} onClick={() => setTab("model")}>Model</button>
          <button className={tab === "sampling" ? "active" : ""} onClick={() => setTab("sampling")}>Sampling</button>
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
        </nav>
      </header>
      <main>
        {tab === "model" && (
          <ModelTab
            models={models}
            info={info}
            layers={layers}
            form={form}
            status={status}
            loading={loading}
            loaded={loaded}
            onRefresh={refreshModels}
            onSelect={selectModel}
            onProbe={probe}
            onForm={updateForm}
            onLayers={setLayers}
            onLoad={load}
            onUnload={unload}
          />
        )}
        {tab === "sampling" && (
          <SamplingTab state={sampling} onChange={(patch) => setSampling((s) => ({ ...s, ...patch }))} />
        )}
        {tab === "chat" && (
          <ChatTab
            messages={messages}
            streaming={streaming}
            status={chatStatus}
            onSend={send}
            onClear={clearChat}
          />
        )}
      </main>
    </div>
  );
}
