---
title: 【接入】GitHub 协作接入 + 门禁验证（10 分钟，先做这个）
labels: server, 接入, P0
assignees: 李
---

## 目标

把工作方式从"发 zip"切到"GitHub 协作"，并**验证门禁真的生效**。做完这一个 Issue，你才有后面所有任务的施工条件。

## 背景（一次真实事故）

10-10 你交付的 `api.py`（T9/T2/T10）**底本停在 9-28 之前**，缺主线已验收的三项修复：

| 主线已有 | 你的文件 |
|---------|:---:|
| N16 坏帧判定归一化（`BAD_NORM_DIM=640`，阈值 20/120/12000） | ❌ 无 |
| P2.9 `timestamp` 必填校验（`_parse_timestamp`） | ❌ 无 |
| P2.10 overlay 规格压缩（`_encode_overlay`，防 1MB/帧） | ❌ 无 |

若直接覆盖会**静默丢失**这三项。所以从此改成：**`main` 是唯一真源，改动一律走分支 + PR**（决策见 `docs/决策记录.md` D15；规范见 `docs/协作规范_GitHub.md`）。

> 你的 T9/T2/T10 **没有丢** —— 已由我方**移植**进主线（PR 待合），合并后你 clone/pull 就能看到。

## 任务（四步，别跳步）

- [ ] **① 接上仓库**（约 5 分钟）

  ```powershell
  # 收到 egomed_20261012.bundle（约 13 MB），放到 D:\ 下
  git clone egomed_20261012.bundle egomed-agent
  cd egomed-agent
  git remote set-url origin https://github.com/KissRain25/egomed-agent-demo.git
  git pull --rebase
  git log --oneline -3        # 应能看到 09cc900 及更早的历史
  ```

- [ ] **② 环境自检**（期望 12 PASS / 0 FAIL）

  ```powershell
  python scripts/check_env.py
  ```

- [ ] **③ 跑门禁**（期望 8/8；这就是以后每次 PR 会自动跑的东西）

  ```powershell
  回归.bat --suite mock
  ```

- [ ] **④ 验证分支保护**（★ 最关键，请如实回报结果）

  ```powershell
  git push origin main
  ```

  | 结果 | 含义 | 你要做的 |
  |------|------|---------|
  | ❌ 被拒（`remote rejected` / `GH006 protected branch`） | ✅ 保护生效 | **立刻在本 Issue 里回报"被拒"+ 报错原文** |
  | ✅ 推成功了 | ⚠️ 配置有漏洞 | **马上停手并告诉闫**（不要继续推；main 会被污染） |

## 验收标准

| 项 | 判据 |
|----|------|
| ① 仓库 | `git log` 能看到历史；`git remote -v` 指向 `KissRain25/egomed-agent-demo` |
| ② 自检 | 12 PASS / 0 FAIL（有 FAIL 就贴出来） |
| ③ 门禁 | mock **8/8** |
| ④ 保护 | 直推 main **被拒**（贴报错原文） |

## 红线

1. **不要直接推 `main`** —— 一切改动走「分支 → PR → 闫 Approve」
2. **不要用你本地那份旧 `api.py` 覆盖仓库** —— 主线已含 N16/P2.9/P2.10
3. 卡住了**当天说**（卡在哪 + 报错原文 + 需要什么），不要默默拖

## 完成后

在 Issue 里贴一段：

```
① clone ✓  HEAD=?
② check_env: ? PASS / ? FAIL
③ 回归 mock: ?/8
④ 直推 main: 被拒 / 成功（报错原文：...）
```

然后就可以接 **Issue #2（T1 并发超时清零）**。

## 参考资料

- `docs/协作规范_GitHub.md`（分支模型 / 日常命令 / 例外流程 / 降级方案）
- `docs/派工单_李_周计划_W2W4_20261012.md`（T1~T8 全周任务与硬期限）
