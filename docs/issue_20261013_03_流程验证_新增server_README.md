---
title: 【测试】流程验证 —— 新增 server/README.md（5 分钟，练手用）
labels: server, 测试
assignees: 李
---

## 目的

这是一个**练手任务**：不是为了修 bug，而是让你（和我）把 GitHub 协作的**整条链路**跑通一次。
做完你就掌握了以后每次干活的标准动作。

**为什么选"新增一个说明文件"**：零风险（新文件、不碰任何代码）、在你负责的目录里（`server/**`）、但**必须走分支 + PR + CI + 我审核** —— 正好把该练的都练到。

## 任务（7 步）

- [ ] **① 同步主分支**

  ```powershell
  git switch main
  git pull --rebase
  git rev-parse --short HEAD        # ★ 记下这个 hash，PR 里要填
  ```

- [ ] **② 建任务分支**（**务必先切分支再动手**）

  ```powershell
  git switch -c server/readme
  ```

- [ ] **③ 新建 `server/README.md`**，写 5~8 行即可，例如：

  ```markdown
  # 服务端（server/）

  - 职责：接收客户端 A1（画面）/ A2（语音），在服务器完成模态识别、检测、分割、ASR、NLU
  - 启动：`python server/api.py`（假实现，秒起） / `python server/api.py --real`（真引擎，预热 60~90s）
  - 接口（契约 v1.0）：`POST /api/v1/frame`、`POST /api/v1/audio`
  - 运维接口：`GET /healthz`（`state` = warming/ready、`rss_mb`、`gpu_mem_mb`）
  - 本文件由李于 2026-10-13 新增，用于验证 GitHub 协作流程（Issue #<本 issue 号>）
  ```

- [ ] **④ 自测**

  ```powershell
  python -m py_compile server/api.py     # 至少语法过
  回归.bat --suite mock                  # 期望 8/8
  ```

- [ ] **⑤ 提交并推分支**

  ```powershell
  git add server/README.md
  git commit -m "docs(server): 新增服务端说明 README（流程验证）"
  git push -u origin server/readme
  ```

- [ ] **⑥ 在 GitHub 上开 PR**

  推送成功后终端会给出链接（`.../pull/new/server/readme`），点进去：
  - **标题**：`docs(server): 新增服务端说明 README（流程验证）`
  - **正文**：用模板逐项填，并在最后加一行 `Closes #<本 issue 号>`（合并后自动关 Issue）
  - 基于 commit 填 ① 记下的 hash

- [ ] **⑦ 等我 Approve**（你会看到 Merge 按钮是灰的 —— 这是**正常**的，说明"必须经审核"生效了）

## 验收标准

| 项 | 判据 |
|----|------|
| 分支 | 你在 `server/readme` 分支上（**没有**直接推 main） |
| 自测 | 回归 mock **8/8** |
| PR | 已开、模板字段填全（含「基于 commit」） |
| CI | PR 上 `mock 8/8（无需 GPU）` 变 **绿** |
| 审核 | 我 Approve 之前**合不了**（Merge 灰） |
| 合并 | 我合并后，本 Issue **自动关闭**（因为写了 `Closes #n`） |

## 红线（这次也照守）

1. **不要直推 main**（会被拒，报 `GH006`；被拒是正常的）
2. 只新增 `server/README.md` —— **不要顺手改别的文件**
3. 卡住了在 Issue 里说（贴报错原文），别自己琢磨太久

## 完成后

在这个 Issue 里贴：

```
① HEAD=?  ② 分支=server/readme  ③ 自测 mock ?/8  ④ PR 链接=?  ⑤ CI=?（绿/红）
遇到的第一个卡点是什么（如果有）：...
```

## 顺手的心得（做完再回来看）

- **`Closes #n`** 是 Issue 与 PR 的"自动关联开关"：以后每个任务都用它，Issue 会自动闭环，不用手动关
- **被拒的推送（GH006）** 不是错误，是**门禁在保护主干** —— 你只需要改用分支 + PR
- 这套流程的价值：**每次改动都有记录、有自测、有审核、有 CI 兜底** —— 再不会出现"我的底本和最新版不一致"那种事
