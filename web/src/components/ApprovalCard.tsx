import type { PendingApproval } from "../state/chatReducer";

type ApprovalCardProps = {
  assistantId: string;
  approvals: PendingApproval[];
  onResolve: (assistantId: string, decision: "approve" | "reject") => void;
};

/**
 * HITL 审批卡片：展示被拦下的工具调用及其参数，
 * 由用户点击“批准并继续 / 拒绝”，再调用 /api/chat/resume 恢复执行。
 */
export function ApprovalCard({ assistantId, approvals, onResolve }: ApprovalCardProps) {
  const actions = approvals.flatMap((item) => item.actions);

  if (actions.length === 0) {
    return null;
  }

  return (
    <section className="panel panel--approval">
      <h4>需要你确认</h4>
      <p className="approval-hint">
        以下工具调用被安全策略拦下，批准后才会真正执行。
      </p>

      <div className="approval-list">
        {actions.map((action, index) => (
          <div className="approval-item" key={`${action.name}-${index}`}>
            <div className="approval-item__head">
              <span className="approval-tool">{action.name}</span>
              {action.description && (
                <span className="approval-desc">{action.description}</span>
              )}
            </div>
            <pre className="approval-args">
              {JSON.stringify(action.args ?? {}, null, 2)}
            </pre>
          </div>
        ))}
      </div>

      <div className="approval-actions">
        <button
          type="button"
          className="approval-approve"
          onClick={() => onResolve(assistantId, "approve")}
        >
          批准并继续
        </button>
        <button
          type="button"
          className="approval-reject"
          onClick={() => onResolve(assistantId, "reject")}
        >
          拒绝
        </button>
      </div>
    </section>
  );
}
