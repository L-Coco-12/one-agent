export type ThinkingEvent = {
  type: "thinking";
  content: string;
  id?: number;
};

export type TextEvent = {
  type: "text";
  content: string;
};

export type ToolCallEvent = {
  type: "tool_call";
  name: string;
  args: Record<string, unknown>;
  id?: string;
};

export type ToolResultEvent = {
  type: "tool_result";
  name: string;
  content: string;
  success?: boolean;
};

export type DoneEvent = {
  type: "done";
  response?: string;
  structured?: Record<string, unknown>;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

/** HITL：被拦下、等待人工审批的一次工具调用 */
export type ActionRequest = {
  name: string;
  args?: Record<string, unknown>;
  description?: string;
};

export type ReviewConfig = {
  action_name?: string;
  allowed_decisions?: string[];
};

/** HITL：一次中断（一次可能包含多个待审批的 action_request） */
export type InterruptRequest = {
  id?: string;
  value?: {
    action_requests?: ActionRequest[];
    review_configs?: ReviewConfig[];
  };
};

export type InterruptEvent = {
  type: "interrupt";
  interrupts: InterruptRequest[];
};

export type AgentStreamEvent =
  | ThinkingEvent
  | TextEvent
  | ToolCallEvent
  | ToolResultEvent
  | DoneEvent
  | ErrorEvent
  | InterruptEvent;
