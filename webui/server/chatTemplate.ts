export interface ContentPart {
  type?: string;
  text?: string;
}

export type MessageContent = string | ContentPart[] | null | undefined;

export interface ToolCall {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string | Record<string, unknown> };
  name?: string;
  arguments?: string | Record<string, unknown>;
}

export interface ChatMessageInput {
  role: string;
  content?: MessageContent;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  reasoning_content?: string;
}

export interface ToolDefinition {
  type?: string;
  function?: { name?: string; description?: string; parameters?: unknown };
}

const TOOL_HEADER =
  "# Tools\n\nYou have access to the following functions:\n\n";
const TOOL_FOOTER =
  "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
  "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
  "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n" +
  "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
  "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
  "- Required parameters MUST be specified\n" +
  "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
  "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
  "</IMPORTANT>";

export function renderContent(content: MessageContent): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const part of content) {
      if (part && typeof part === "object" && typeof part.text === "string") out += part.text;
    }
    return out;
  }
  return String(content);
}

function formatArg(value: unknown): string {
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function splitReasoning(content: string): { reasoning: string; visible: string } {
  const marker = "</think>";
  if (!content.includes(marker)) return { reasoning: "", visible: content };
  const head = content.slice(0, content.indexOf(marker));
  const tail = content.slice(content.indexOf(marker) + marker.length);
  const inner = head.includes("<think>") ? head.slice(head.lastIndexOf("<think>") + "<think>".length) : head;
  return { reasoning: inner.trim(), visible: tail.replace(/^\s+/, "") };
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

export class ReasoningSplitter {
  private buf = "";
  private mode: "probe" | "reasoning" | "content" = "probe";
  private started = false;

  constructor(private expecting: boolean) {}

  private lead(text: string): string {
    if (this.started) return text;
    this.started = true;
    return text.replace(/^\s+/, "");
  }

  push(delta: string): { reasoning: string; content: string } {
    if (this.mode === "content") return { reasoning: "", content: delta };
    this.buf += delta;
    if (this.mode === "probe") {
      if (this.buf.startsWith(THINK_OPEN)) {
        this.buf = this.buf.slice(THINK_OPEN.length);
        this.mode = "reasoning";
      } else if (this.expecting) {
        const idx = this.buf.indexOf(THINK_CLOSE);
        if (idx < 0) return { reasoning: "", content: "" };
        const reasoning = this.lead(this.buf.slice(0, idx));
        const content = this.buf.slice(idx + THINK_CLOSE.length).replace(/^\s+/, "");
        this.buf = "";
        this.mode = "content";
        return { reasoning, content };
      } else {
        if (this.buf.length < 8) return { reasoning: "", content: "" };
        this.mode = "content";
        const out = this.buf;
        this.buf = "";
        return { reasoning: "", content: out };
      }
    }
    const idx = this.buf.indexOf(THINK_CLOSE);
    if (idx < 0) {
      const keep = THINK_CLOSE.length - 1;
      if (this.buf.length <= keep) return { reasoning: "", content: "" };
      const out = this.lead(this.buf.slice(0, this.buf.length - keep));
      this.buf = this.buf.slice(this.buf.length - keep);
      return { reasoning: out, content: "" };
    }
    const reasoning = this.lead(this.buf.slice(0, idx));
    const content = this.buf.slice(idx + THINK_CLOSE.length).replace(/^\s+/, "");
    this.buf = "";
    this.mode = "content";
    return { reasoning, content };
  }

  flush(): { reasoning: string; content: string } {
    const rest = this.buf;
    this.buf = "";
    if (this.mode === "reasoning") {
      const idx = rest.indexOf(THINK_CLOSE);
      if (idx < 0) return { reasoning: this.lead(rest), content: "" };
      return {
        reasoning: this.lead(rest.slice(0, idx)),
        content: rest.slice(idx + THINK_CLOSE.length).replace(/^\s+/, ""),
      };
    }
    if (this.mode === "probe") {
      if (rest.startsWith(THINK_OPEN)) return { reasoning: this.lead(rest.slice(THINK_OPEN.length)), content: "" };
      if (this.expecting) return { reasoning: this.lead(rest), content: "" };
      return { reasoning: "", content: rest };
    }
    return { reasoning: "", content: rest };
  }
}

function toolsSystemBlock(tools: ToolDefinition[], systemContent: string, toolHint: string): string {
  let out = "<|im_start|>system\n" + TOOL_HEADER + "<tools>";
  for (const tool of tools) out += "\n" + JSON.stringify(tool);
  out += "\n</tools>" + TOOL_FOOTER;
  if (toolHint) out += "\n\n" + toolHint;
  if (systemContent) out += "\n\n" + systemContent;
  return out + "<|im_end|>\n";
}

function toolCallText(out: string, visible: string, calls: ToolCall[]): string {
  for (let idx = 0; idx < calls.length; idx++) {
    const call = calls[idx];
    const fn = call.function ?? call;
    const name = fn.name ?? "";
    if (idx === 0) {
      out += visible.trim() ? "\n\n<tool_call>\n<function=" + name + ">\n" : "<tool_call>\n<function=" + name + ">\n";
    } else {
      out += "\n<tool_call>\n<function=" + name + ">\n";
    }
    let parsed: unknown = fn.arguments;
    if (typeof parsed === "string") {
      try {
        parsed = JSON.parse(parsed);
      } catch {
        parsed = undefined;
      }
    }
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        out += "<parameter=" + key + ">\n" + formatArg(value) + "\n</parameter>\n";
      }
    }
    out += "</function>\n</tool_call>";
  }
  return out;
}

export function buildQwenPrompt(
  messages: ChatMessageInput[],
  tools: ToolDefinition[],
  enableThinking: boolean,
  toolHint = "",
): string {
  const msgs = messages.filter((m) => m && typeof m.role === "string");
  if (!msgs.length) throw new Error("no messages provided");

  const first = msgs[0];
  const systemContent = first.role === "system" ? renderContent(first.content).trim() : "";

  let out = "";
  if (tools.length > 0) {
    out += toolsSystemBlock(tools, systemContent, toolHint);
  } else if (first.role === "system") {
    out += "<|im_start|>system\n" + systemContent + "<|im_end|>\n";
  }

  let lastQuery = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role !== "user") continue;
    const text = renderContent(msgs[i].content).trim();
    if (!(text.startsWith("<tool_response>") && text.endsWith("</tool_response>"))) {
      lastQuery = i;
      break;
    }
  }

  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    const content = renderContent(m.content).trim();
    if (m.role === "system") continue;

    if (m.role === "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role === "assistant") {
      let reasoning = typeof m.reasoning_content === "string" ? m.reasoning_content.trim() : "";
      let visible = content;
      if (!reasoning) {
        const split = splitReasoning(content);
        reasoning = split.reasoning;
        visible = split.visible;
      }
      out += "<|im_start|>assistant\n";
      out += i > lastQuery ? "<think>\n" + reasoning + "\n</think>\n\n" + visible : visible;
      if (Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
        out = toolCallText(out, visible, m.tool_calls);
      }
      out += "<|im_end|>\n";
    } else if (m.role === "tool") {
      const prevTool = i > 0 && msgs[i - 1].role === "tool";
      const nextTool = i + 1 < msgs.length && msgs[i + 1].role === "tool";
      if (!prevTool) out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (!nextTool) out += "<|im_end|>\n";
    }
  }

  out += "<|im_start|>assistant\n";
  out += enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return out;
}
