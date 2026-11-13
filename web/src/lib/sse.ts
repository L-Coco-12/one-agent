import type { AgentStreamEvent } from "../types/events";

const STREAM_EVENT_TYPES = [
  "thinking",
  "text",
  "tool_call",
  "tool_result",
  "done",
  "interrupt",
  "agent_error",
] as const;

type StreamOptions = {
  apiBaseUrl: string;
  message: string;
  threadId: string;
  onEvent: (event: AgentStreamEvent) => void;
  onError: (error: string) => void;
};

type ResumeOptions = {
  apiBaseUrl: string;
  threadId: string;
  decisions: Record<string, unknown>[];
  onEvent: (event: AgentStreamEvent) => void;
  onError: (error: string) => void;
};

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
}

function isTerminal(event: AgentStreamEvent): boolean {
  return (
    event.type === "done" || event.type === "error" || event.type === "interrupt"
  );
}

export function openChatStream({
  apiBaseUrl,
  message,
  threadId,
  onEvent,
  onError,
}: StreamOptions): () => void {
  const endpoint = new URL(`${normalizeBaseUrl(apiBaseUrl)}/api/chat/stream`);
  endpoint.searchParams.set("message", message);
  endpoint.searchParams.set("thread_id", threadId);

  const source = new EventSource(endpoint.toString());
  let terminalEventHandled = false;

  for (const eventName of STREAM_EVENT_TYPES) {
    source.addEventListener(eventName, (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent<string>).data) as AgentStreamEvent;
        onEvent(payload);

        // HITL 中断：本轮到此为止，主动关闭 EventSource，
        // 否则浏览器会自动重连并重新执行同一句话。
        if (isTerminal(payload)) {
          terminalEventHandled = true;
          source.close();
        }
      } catch (err) {
        const messageFromError = err instanceof Error ? err.message : String(err);
        onError(`解析 SSE 数据失败：${messageFromError}`);
        source.close();
      }
    });
  }

  source.onerror = () => {
    if (terminalEventHandled) {
      return;
    }
    onError("SSE 连接失败。");
    source.close();
  };

  return () => {
    source.close();
  };
}

/**
 * 人工审批后恢复执行。
 *
 * 浏览器 EventSource 只能发 GET，所以这里用 fetch POST + 手写 SSE 解析。
 */
export function openResumeStream({
  apiBaseUrl,
  threadId,
  decisions,
  onEvent,
  onError,
}: ResumeOptions): () => void {
  const controller = new AbortController();
  let closed = false;
  let terminal = false;

  const run = async () => {
    try {
      const response = await fetch(
        `${normalizeBaseUrl(apiBaseUrl)}/api/chat/resume`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ thread_id: threadId, decisions }),
          signal: controller.signal,
        },
      );

      if (!response.ok || !response.body) {
        throw new Error(`恢复执行失败（${response.status}）`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const payload = parseSseFrame(frame);
          if (payload) {
            onEvent(payload);
            if (isTerminal(payload)) {
              terminal = true;
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }

      if (!terminal && !closed && !controller.signal.aborted) {
        onError("SSE 连接在完成前关闭。");
      }
    } catch (error) {
      if (closed || controller.signal.aborted) {
        return;
      }
      const message = error instanceof Error ? error.message : String(error);
      onError(message);
    }
  };

  void run();

  return () => {
    closed = true;
    controller.abort();
  };
}

function parseSseFrame(frame: string): AgentStreamEvent | null {
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return JSON.parse(dataLines.join("\n")) as AgentStreamEvent;
  } catch {
    return null;
  }
}
