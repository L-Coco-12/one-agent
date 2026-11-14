# RAG 检索评估报告

- 生成时间：2026-11-14 05:59:23Z
- 问题集：`D:\deepseekcode\skills-agent-proto\evals\rag_questions.jsonl`（50 题）
- collection：`knowledge`，top_k=6
- 指标口径：Hit@k 要求期望 section 出现在前 k 条（多跳题要求全部 section）；MRR/Precision@k 只统计单跳题；无答案题看 top-1 相似度是否低于阈值。

## 汇总

| 指标 | 值 |
| --- | --- |
| Hit@1 | 70.0% |
| Hit@4 | 92.5% |
| MRR（单跳） | 0.882 |
| Precision@k（单跳） | 0.294 |
| 多跳全覆盖率 | 100.0% |
| 无答案拒答率 | 0.0% |
| 无答案平均 top-1 score | 0.645（阈值 0.45） |

## 逐题明细

| id | 类型 | 命中排名 | top-1 位置 | top-1 score | 期望 section |
| --- | --- | --- | --- | --- | --- |
| q01 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 七天无理由退货 | 0.724 | 退换货政策常见问题 > 七天无理由退货 |
| q02 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 七天无理由退货 | 0.644 | 退换货政策常见问题 > 七天无理由退货 |
| q03 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.626 | 退换货政策常见问题 > 退货流程 |
| q04 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.734 | 退换货政策常见问题 > 退货流程 |
| q05 | single_hop | 1 | product/faq.md > 退换货政策常见问题 > 退货流程 | 0.721 | 退换货政策常见问题 > 退货流程 |
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
| q21 | single_hop | 3 | fastapi/tutorial-query-params-str-validations.md > 查询参数和字符串校验 > 在 `q` 的 `Annotated` 中添加 `Query` | 0.794 | 查询参数和字符串校验 > 添加更多校验 |
| q22 | single_hop | 1 | fastapi/tutorial-query-params-str-validations.md > 查询参数和字符串校验 > 添加正则表达式 | 0.714 | 查询参数和字符串校验 > 添加正则表达式 |
| q23 | single_hop | 未命中 | fastapi/tutorial-path-params.md > 路径参数 > 数据校验 | 0.712 | 路径参数和数值校验 > 数值校验：大于和小于等于 |
| q24 | single_hop | 1 | fastapi/tutorial-response-model.md > 响应模型 - 返回类型 > `response_model` 参数 > `response_model` 的优先级 | 0.819 | 响应模型 - 返回类型 > `response_model` 参数 |
| q25 | single_hop | 2 | fastapi/tutorial-response-model.md > 响应模型 - 返回类型 > 响应模型的编码参数 > `response_model_include` 和 `response_model_exclude` | 0.714 | 响应模型 - 返回类型 > 响应模型的编码参数 > 使用 `response_model_exclude_unset` 参数 |
| q26 | single_hop | 3 | fastapi/advanced-custom-response.md > 自定义响应 - HTML、流、文件等 > 可用响应 > `Response` | 0.733 | 处理错误 > 添加自定义响应头 |
| q27 | single_hop | 1 | fastapi/tutorial-middleware.md > 中间件 > 多个中间件的执行顺序 | 0.722 | 中间件 > 多个中间件的执行顺序 |
| q28 | single_hop | 1 | fastapi/tutorial-cors.md > CORS（跨域资源共享） > 使用 `CORSMiddleware` | 0.787 | CORS（跨域资源共享） > 使用 `CORSMiddleware` |
| q29 | single_hop | 1 | fastapi/tutorial-testing.md > 测试 > 使用 `TestClient` | 0.748 | 测试 > 使用 `TestClient` |
| q30 | single_hop | 1 | fastapi/advanced-events.md > 生命周期事件 > Lifespan | 0.645 | 生命周期事件 > Lifespan |
| q31 | single_hop | 1 | fastapi/advanced-settings.md > 设置和环境变量 > 读取 `.env` 文件 > 从 `.env` 中读取设置 | 0.810 | 设置和环境变量 > 读取 `.env` 文件 |
| q32 | single_hop | 2 | fastapi/advanced-websockets.md > WebSockets > 尝试一下 | 0.785 | WebSockets > 创建 `websocket` |
| q33 | single_hop | 1 | kubernetes/concepts-workloads-pods-probes.md > 探针的类型 | 0.849 | 探针的类型 |
| q34 | single_hop | 3 | kubernetes/concepts-workloads-pods-probes.md > 探针结果 | 0.787 | 探针的类型 > 存活探针 |
| q35 | single_hop | 1 | kubernetes/concepts-workloads-pods-probes.md > 探针的类型 > 就绪探针 | 0.765 | 探针的类型 > 就绪探针 |
| q36 | single_hop | 1 | kubernetes/concepts-workloads-pods-pod-lifecycle.md > Pod 如何处理容器问题 > Pod 级别容器重启策略 | 0.791 | Pod 如何处理容器问题 > Pod 级别容器重启策略 |
| q37 | single_hop | 1 | kubernetes/concepts-workloads-controllers-deployment.md > 回滚 Deployment > 检查 Deployment 上线历史 | 0.744 | 回滚 Deployment > 检查 Deployment 上线历史 |
| q38 | single_hop | 1 | kubernetes/concepts-services-networking-service.md > Service 类型 | 0.775 | Service 类型 |
| q39 | multi_hop | 6 | fastapi/tutorial-response-model.md > 响应模型 - 返回类型 > 响应模型的编码参数 > `response_model_include` 和 `response_model_exclude` | 0.799 | 响应模型 - 返回类型 > `response_model` 参数 | 响应模型 - 返回类型 > 响应模型的编码参数 > 使用 `response_model_exclude_unset` 参数 |
| q40 | multi_hop | 6 | fastapi/tutorial-middleware.md > 中间件 > 多个中间件的执行顺序 | 0.755 | 中间件 > 多个中间件的执行顺序 | CORS（跨域资源共享） > 使用 `CORSMiddleware` > CORS 预检请求 |
| q41 | multi_hop | 2 | kubernetes/concepts-workloads-pods-probes.md > 探针结果 | 0.801 | 探针的类型 > 存活探针 | 探针的类型 > 就绪探针 |
| q42 | multi_hop | 2 | kubernetes/concepts-workloads-controllers-deployment.md > 编写 Deployment 规约 > 进度期限秒数 | 0.778 | 编写 Deployment 规约 > 进度期限秒数 | 编写 Deployment 规约 > 最短就绪时间 |
| q43 | no_answer | 拒答失败 | fastapi/tutorial-first-steps.md > 第一步 | 0.664 | （知识库未覆盖） |
| q44 | no_answer | 拒答失败 | fastapi/tutorial-first-steps.md > 第一步 | 0.694 | （知识库未覆盖） |
| q45 | no_answer | 拒答失败 | fastapi/tutorial-first-steps.md > 第一步 | 0.643 | （知识库未覆盖） |
| q46 | no_answer | 拒答失败 | kubernetes/concepts-workloads-pods-pod-lifecycle.md > Pod 如何处理容器问题 | 0.648 | （知识库未覆盖） |
| q47 | no_answer | 拒答失败 | kubernetes/concepts-workloads-controllers-deployment.md > 缩放 Deployment | 0.715 | （知识库未覆盖） |
| q48 | no_answer | 拒答失败 | kubernetes/concepts-services-networking-service.md > Kubernetes 中的 Service | 0.724 | （知识库未覆盖） |
| q49 | no_answer | 拒答失败 | product/faq.md > 退换货政策常见问题 > 换货规则 | 0.564 | （知识库未覆盖） |
| q50 | no_answer | 拒答失败 | product/faq.md > 退换货政策常见问题 > 发票相关 | 0.536 | （知识库未覆盖） |

## 失败分析

- **q23**（FastAPI 路径参数 item_id 要限制成大于 0 且小于等于 10，代码里怎么声明？）：期望 section 未出现在 top-k
- **q19**（支持微信或支付宝原路退款吗？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q20**（海外订单怎么退货？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q43**（FastAPI 怎么接入 Redis 做响应缓存？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q44**（FastAPI 里怎么用 Celery 跑异步任务队列？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q45**（FastAPI 怎么实现定时任务调度？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q46**（Kubernetes 的 etcd 数据怎么做备份和恢复？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q47**（Kubernetes 的水平自动扩缩容 HPA 依据哪些指标扩缩 Pod？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q48**（Kubernetes 里怎么配置 Istio 服务网格做流量治理？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q49**（我们平台的会员积分怎么兑换优惠券？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
- **q50**（公司差旅报销的流程和审批时限是什么？）：无答案题但 top-1 相似度过高，模型可能误以为知识库有答案
