import { useState } from "react";
import type { ChatMessage } from "../../server/types";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  status: string;
  thinking: boolean;
  hideThinking: boolean;
  onOptions: (patch: { thinking?: boolean; hideThinking?: boolean }) => void;
  onSend: (text: string) => void;
  onStop: () => void;
  onClear: () => void;
}

export function visible(text: string, hide: boolean, thinking: boolean): string {
  if (!hide) return text;
  const closed = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const open = closed.indexOf("<think>");
  if (open >= 0) return closed.slice(0, open);
  const end = closed.indexOf("</think>");
  if (end >= 0) return closed.slice(end + "</think>".length);
  return thinking ? "" : closed;
}

export function ChatTab({ messages, streaming, status, thinking, hideThinking, onOptions, onSend, onStop, onClear }: Props) {
  const [input, setInput] = useState("");

  const send = () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="chat">
      <div className="row chat-options">
        <label className="check">
          <input type="checkbox" checked={thinking}
            onChange={(e) => onOptions({ thinking: e.target.checked })} />
          Thinking
        </label>
        <label className="check">
          <input type="checkbox" checked={hideThinking}
            onChange={(e) => onOptions({ hideThinking: e.target.checked })} />
          Hide thinking in chat
        </label>
      </div>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <span className="role">{msg.role}</span>
            <div className="content">
              {msg.role === "assistant" ? visible(msg.content, hideThinking, thinking) : msg.content}
            </div>
          </div>
        ))}
        {!messages.length && <div className="muted">No messages</div>}
      </div>
      <div className="composer">
        <textarea
          rows={2}
          placeholder="Message"
          value={input}
          disabled={streaming}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="composer-actions">
          {streaming ? (
            <button className="stop" onClick={onStop}>Stop</button>
          ) : (
            <button onClick={send} disabled={!input.trim()}>Send</button>
          )}
          <button className="secondary" onClick={onClear} disabled={streaming}>Clear</button>
        </div>
      </div>
      <div className="status">{status}</div>
    </div>
  );
}
