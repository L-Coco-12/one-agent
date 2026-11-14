---
title: Cookie 参数
updated_at: "2026-11-14"
owner: FastAPI 中文文档（社区翻译）
tags: [fastapi, python, web]
source_url: "https://github.com/fastapi/fastapi/blob/master/docs/zh/docs/tutorial/cookie-params.md"
license: MIT
---
# Cookie 参数
定义 `Cookie` 参数与定义 `Query` 和 `Path` 参数一样。

## 导入 `Cookie`
首先，导入 `Cookie`：

## 声明 `Cookie` 参数
声明 `Cookie` 参数的方式与声明 `Query` 和 `Path` 参数相同。

你可以定义默认值，以及所有额外的验证或注解参数：

`Cookie` 、`Path` 、`Query` 是**兄弟类**，都继承自共用的 `Param` 类。

注意，从 `fastapi` 导入的 `Query`、`Path`、`Cookie` 等对象，实际上是返回特殊类的函数。

必须使用 `Cookie` 声明 cookie 参数，否则该参数会被解释为查询参数。

请注意，由于**浏览器会以特殊方式并在幕后处理 cookies**，它们**不会**轻易允许**JavaScript**访问它们。

如果你前往位于 `/docs` 的**API 文档界面**，你可以看到你的*路径操作*中有关 cookies 的**文档**。

但即使你**填写了数据**并点击 "Execute"，由于文档界面依赖于**JavaScript**工作，cookies 也不会被发送，你会看到一个**错误**消息，好像你没有填写任何值一样。

## 小结
使用 `Cookie` 声明 cookie 参数的方式与 `Query` 和 `Path` 相同。
