# 知识库来源与许可

本目录中的文档用于 RAG 学习与评估，来源如下。若你是这些文档的版权方并希望移除，
请提 issue，我们会删除对应文件。

## 1. 示例业务文档（合成）

| 路径 | 说明 |
| --- | --- |
| `product/faq.md` | 虚构电商售后 FAQ，为本项目自写示例 |
| `api/auth.md` | 虚构内部 API 鉴权说明，为本项目自写示例 |

## 2. FastAPI 中文文档

- 来源仓库：<https://github.com/fastapi/fastapi>（`docs/zh/docs` 目录）
- 项目主页：<https://fastapi.tiangolo.com/zh/>
- 许可协议：MIT License（Copyright © 2018 Sebastián Ramírez）
- 收录范围：tutorial 与 advanced 共 22 篇，见 `knowledge/fastapi/` 下文件的
  `source_url` 字段（每篇都指向原始文件）。
- 获取日期：2026-11-14
- 处理方式：仅去除 frontmatter、mkdocs 块标记与文档包含占位符，正文未改写。

## 3. Kubernetes 中文文档

- 来源仓库：<https://github.com/kubernetes/website>（`content/zh-cn/docs` 目录）
- 项目主页：<https://kubernetes.io/zh-cn/docs/>
- 许可协议：CC BY 4.0（© The Kubernetes Authors，详见
  <https://github.com/kubernetes/website/blob/main/LICENSE>）
- 收录范围：concepts 下 12 篇，见 `knowledge/kubernetes/` 下文件的 `source_url` 字段。
- 获取日期：2026-11-14
- 处理方式：仅去除 frontmatter、Hugo shortcode 与 HTML 注释，正文未改写。

## 使用提醒

- CC BY 4.0 要求署名：对外展示知识库内容时保留上述来源与作者信息。
- 这些文档仅用于本地学习/评估，未用于商业分发。
- 重新生成索引：`uv run python scripts/build_rag_index.py --force`。
