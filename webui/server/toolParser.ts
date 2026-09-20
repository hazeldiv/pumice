export interface ParsedToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

function coerce(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const first = trimmed[0];
  if (first === "{" || first === "[" || first === '"' || trimmed === "true" || trimmed === "false" || trimmed === "null") {
    try {
      return JSON.parse(trimmed);
    } catch {
      return trimmed;
    }
  }
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) return Number(trimmed);
  return trimmed;
}

function parseXmlBlock(inner: string): { name: string; args: Record<string, unknown> } | null {
  const fn = /<function=([^>\s]+)\s*>/.exec(inner);
  if (!fn) return null;
  const args: Record<string, unknown> = {};
  const re = /<parameter=([^>\s]+)\s*>\n?([\s\S]*?)\n?<\/parameter>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(inner)) !== null) args[m[1]] = coerce(m[2]);
  return { name: fn[1], args };
}

function parseJsonBlock(inner: string): { name: string; args: Record<string, unknown> } | null {
  let parsed: any;
  try {
    parsed = JSON.parse(inner.trim());
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const fn = parsed.function ?? parsed;
  const name = String(fn?.name ?? "");
  if (!name) return null;
  let args = fn?.arguments ?? {};
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = {};
    }
  }
  return { name, args: args && typeof args === "object" ? args : {} };
}

let callSeq = 0;

export function parseToolCalls(text: string): { content: string; toolCalls: ParsedToolCall[] } {
  const toolCalls: ParsedToolCall[] = [];
  const re = /<tool_call>([\s\S]*?)<\/tool_call>/g;
  let content = "";
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    content += text.slice(last, m.index);
    last = m.index + m[0].length;
    const inner = m[1];
    const parsed = inner.trim().startsWith("{") ? parseJsonBlock(inner) : parseXmlBlock(inner);
    if (parsed) {
      toolCalls.push({
        id: `call_${Date.now().toString(36)}_${callSeq++}`,
        type: "function",
        function: { name: parsed.name, arguments: JSON.stringify(parsed.args) },
      });
    } else {
      content += m[0];
    }
  }
  content += text.slice(last);
  return { content: content.trim(), toolCalls };
}
