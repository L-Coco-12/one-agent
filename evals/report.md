# RAG 检索评估报告

- 生成时间：2026-11-14 04:22:05Z
- 问题集：`D:\deepseekcode\skills-agent-proto\evals\rag_questions.jsonl`（20 题）
- collection：`knowledge`，top_k=6
- 指标口径：Hit@k 要求期望 section 出现在前 k 条（多跳题要求全部 section）；MRR/Precision@k 只统计单跳题；无答案题看 top-1 相似度是否低于阈值。

## 汇总

| 指标 | 值 |
| --- | --- |
| Hit@1 | 88.9% |
| Hit@4 | 100.0% |
| MRR（单跳） | 1.000 |
| Precision@k（单跳） | 0.167 |
| 多跳全覆盖率 | 100.0% |
| 无答案拒答率 | 0.0% |
| 无答案平均 top-1 score | 0.629（阈值 0.45） |

## 逐题明细

| id | 类型 | 命中排名 | top-1 位置 | top-1 score | 期望 section |
| --- | --- | --- | --- | --- | --- |
| q01 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 七天无理由退货 | 0.724 | 退换货政策常见问题 > 七天无理由退货 |
| q02 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 七天无理由退货 | 0.644 | 退换货政策常见问题 > 七天无理由退货 |
| q03 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.626 | 退换货政策常见问题 > 退货流程 |
| q04 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.734 | 退换货政策常见问题 > 退货流程 |
| q05 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.722 | 退换货政策常见问题 > 退货流程 |
| q06 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.683 | 退换货政策常见问题 > 退货流程 |
| q07 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 换货规则 | 0.714 | 退换货政策常见问题 > 换货规则 |
| q08 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 换货规则 | 0.704 | 退换货政策常见问题 > 换货规则 |
| q09 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 换货规则 | 0.665 | 退换货政策常见问题 > 换货规则 |
| q10 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 发票相关 | 0.804 | 退换货政策常见问题 > 发票相关 |
| q11 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 发票相关 | 0.766 | 退换货政策常见问题 > 发票相关 |
| q12 | single_hop | 1 | api/auth.md > API 鉴权说明 > 令牌有效期 | 0.709 | API 鉴权说明 > 令牌有效期 |
| q13 | single_hop | 1 | api/auth.md > API 鉴权说明 > 令牌有效期 | 0.689 | API 鉴权说明 > 令牌有效期 |
| q14 | single_hop | 1 | api/auth.md > API 鉴权说明 > 令牌有效期 | 0.716 | API 鉴权说明 > 令牌有效期 |
| q15 | single_hop | 1 | api/auth.md > API 鉴权说明 > 常见错误码 | 0.750 | API 鉴权说明 > 常见错误码 |
| q16 | single_hop | 1 | api/auth.md > API 鉴权说明 > 获取访问令牌 | 0.749 | API 鉴权说明 > 获取访问令牌 |
| q17 | multi_hop | 2 | api/auth.md > API 鉴权说明 > 令牌有效期 | 0.665 | API 鉴权说明 > 令牌有效期 | API 鉴权说明 > 常见错误码 |
| q18 | multi_hop | 2 | product/faq.md > 退换货政策常见问题 > 发票相关 | 0.767 | 退换货政策常见问题 > 退货流程 | 退换货政策常见问题 > 发票相关 |
| q19 | no_answer | 拒答失败 | product/faq.md > 退换货政策常见问题 > 发票相关 | 0.645 | （知识库未覆盖） |
| q20 | no_answer | 拒答失败 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.613 | （知识库未覆盖） |

## 失败分析

- **q19**（支持微信或支付宝原路退款吗？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q20**（海外订单怎么退货？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
