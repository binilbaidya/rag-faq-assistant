"use client";

// Drop into a Next.js site and set NEXT_PUBLIC_CHAT_API_URL to the deployed API's URL.
import { useState, type FormEvent } from "react";

type Message = { role: "user" | "assistant"; text: string };

const API_URL = process.env.NEXT_PUBLIC_CHAT_API_URL;

export default function ChatWidget() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const append = (text: string) =>
    setMessages((m) => [...m.slice(0, -1), { role: "assistant", text: m[m.length - 1].text + text }]);

  async function send(e: FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }, { role: "assistant", text: "" }]);

    try {
      const res = await fetch(`${API_URL}/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
      });
      if (res.status === 429) throw new Error("Demo limit reached. Please try again later.");
      if (!res.ok || !res.body) throw new Error("Something went wrong. Please try again.");

      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const raw of events) {
          const event = raw.match(/^event: (.*)$/m)?.[1];
          const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? "{}");
          if (event === "token" || event === "error") append(data.message);
        }
      }
    } catch (err) {
      append(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section style={{ maxWidth: 560, border: "1px solid #333", borderRadius: 12, padding: 16, background: "#111", color: "#ededed" }}>
      <h2 style={{ margin: "0 0 4px", fontSize: 18 }}>Ask about me</h2>
      <p style={{ margin: "0 0 12px", fontSize: 13, color: "#999" }}>
        Answers come only from my ingested FAQ. The first reply can take ~30s while the free server wakes up.
      </p>
      <div aria-live="polite" style={{ display: "grid", gap: 8, maxHeight: 360, overflowY: "auto", marginBottom: 12 }}>
        {messages.map((m, i) => (
          <p
            key={i}
            style={{
              margin: 0,
              padding: "8px 12px",
              borderRadius: 8,
              whiteSpace: "pre-wrap",
              justifySelf: m.role === "user" ? "end" : "start",
              background: m.role === "user" ? "#6366f1" : "#1f1f1f",
            }}
          >
            {m.text || "…"}
          </p>
        ))}
      </div>
      <form onSubmit={send} style={{ display: "flex", gap: 8 }}>
        <input
          aria-label="Your question"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          maxLength={2000}
          placeholder="e.g. What does Binil work with?"
          style={{ flex: 1, padding: "8px 12px", borderRadius: 8, border: "1px solid #333", background: "#0a0a0a", color: "#ededed" }}
        />
        <button disabled={busy} style={{ padding: "8px 16px", borderRadius: 8, border: 0, background: "#6366f1", color: "#fff" }}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
