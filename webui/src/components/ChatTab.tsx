import { useState } from "react";
import type { ChatMessage } from "../../server/types";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  status: string;
  onSend: (text: string) => void;
  onClear: () => void;
}

export function ChatTab({ messages, streaming, status, onSend, onClear }: Props) {
  const [input, setInput] = useState("");

  const send = () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <span className="role">{msg.role}</span>
            <div className="content">{msg.content}</div>
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
          <button onClick={send} disabled={streaming || !input.trim()}>Send</button>
          <button className="secondary" onClick={onClear} disabled={streaming}>Clear</button>
        </div>
      </div>
      <div className="status">{status}</div>
    </div>
  );
}
