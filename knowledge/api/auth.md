---
title: API 鉴权说明
updated_at: 2026-10-20
owner: 平台组
tags: [api, auth]
---

# API 鉴权说明

## 获取访问令牌

所有业务接口都需要在请求头中携带 `Authorization: Bearer <token>`。
令牌通过 `POST /api/v1/auth/token` 获取，请求体包含 `client_id` 与 `client_secret`。

## 令牌有效期

- 访问令牌有效期 2 小时；
- 刷新令牌有效期 30 天，到期后需要重新授权；
- 令牌即将过期时，响应头会返回 `X-Token-Expires-In`，单位为秒。

## 常见错误码

| 错误码 | 含义 | 处理方式 |
| --- | --- | --- |
| 40101 | 令牌缺失 | 检查 Authorization 请求头 |
| 40102 | 令牌过期 | 使用刷新令牌换取新令牌 |
| 40301 | 权限不足 | 确认应用是否申请了对应 scope |

## 安全建议

不要把 `client_secret` 写进前端代码或提交到代码仓库；
服务端调用应使用独立的服务账号，并按最小权限原则申请 scope。
