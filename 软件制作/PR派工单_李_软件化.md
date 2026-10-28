# 李的任务：PR 派工单（N53 服务端配合项）

> 创建：2026-10-28 ｜ 维护：闫 ｜ 配套：`软件制作/分工_软件化.md`、`docs/协作规范_GitHub.md`（v1, D15）
> 远端：`https://github.com/KissRain25/egomed-agent-demo.git` ｜ **基线分支：`main`**（本仓库不用 develop）
> 用法：每个 PR 的「标题 / 描述」可直接复制到 GitHub 新建 PR 时粘贴；**「基于 commit」一栏由李开工时填**。

---

## 一、总览（5 个 PR，按依赖排序）

| # | 分支 | PR 标题 | 触碰接口 | 依赖 | 优先级 | 合并条件 |
|---|---|---|---|---|---|---|
| **P1** | `server/n53-1-healthz` | feat(server): N53-1 `/healthz` 增 version 与就绪标志 | **是**（只加字段，v1.0.4） | 无 | **最高**（闫的 demo 卡它） | CI 绿 + 闫确认字段 + 契约同步 |
| **P2** | `server/n53-2-graceful-exit` | fix(server): N53-2 服务端优雅退出，清理子进程与端口 | 否 | 无 | 高 | CI 绿 + 闫实测无残留 |
| **P3** | `server/n53-4-err-codes` | feat(server): N53-4 E2xxx 错误码与统一提示文案 | **是**（错误码/文案，v1.0.4） | P1（同文件建议串行） | 中 | CI 绿 + 双方同意 + 契约同步 |
| **P4** | `server/n53-3-config-list` | docs(server): N53-3 服务端可配置项清单 + 落地 3 项 | 否 | 无 | 中 | CI 绿 + 闫并入 `config.toml` |
| **P5** | `server/n46-soak-guard` | feat(server): N46 长跑守护与观测指标（配合 N53-5 长稳） | 否 | P2 | 中 | 连续 2h 无中断 + 指标留档 |

**串行提醒**：P1 / P2 / P3 都会动 `server/api.py`，**按 P1 → P2 → P3 顺序合并**，每个 PR 从最新 `main` 开分支（≤2 天生命周期），避免三方冲突。

---

## 二、P1：`/healthz` 增 version 与就绪标志（最高优先）

**分支**：`server/n53-1-healthz`

**PR 描述（可直接粘贴）**

```markdown
## 任务
N53-1 服务端健康检查语义固化：`/healthz` 增加 `version` 与就绪标志，供客户端启动器判断"可以开界面了"

## 基线（★ 必填）
- 基于 commit：`<git rev-parse --short HEAD>`
- 分支：`server/n53-1-healthz`

## 改动
- 文件：`server/api.py`、`docs/API_CONTRACT.md`、`docs/接口.md`
- 摘要（≤5 行）：
  1. `/healthz` 响应新增 `version`（服务端版本号）
  2. 新增 `ready`（true=模型预热完成，可接 A1）
  3. 新增 `warmup_ms`（本次启动预热耗时，便于客户端展示"正在预热…"）
  4. 预热未完成时 `ready=false`，A1 仍按原语义返回，不改现有行为
  5. 契约只加字段，走 v1.0.4 增补

## 是否触碰接口
- [x] 是 → 契约变更说明：仅新增 3 个只读字段，不改字段名与语义；**已通知闫**

## 验收（必须有数字）
- 命令：`curl http://127.0.0.1:8000/healthz`
- 数字（改前 → 改后）：响应字段数 2 → 5；预热完成时间 ≈78s（与 N17 一致，偏差 <10%）
- 日志片段：`[warmup] ready=true after 78.2s`
- 补充：`回归.bat --suite mock` 8/8 PASS

## 遗留 / 风险 / 需要对方提供
- 客户端 `launcher.py demo` 将改为轮询 `/healthz` 的 `ready` 字段（替代现在的"日志标记"判据）
```

**为什么最优先**：`launcher.py demo` 要等"服务端真就绪"才开界面。现在只能靠日志标记猜，字段化之后才算可靠。

---

## 三、P2：优雅退出

**分支**：`server/n53-2-graceful-exit`

```markdown
## 任务
N53-2 服务端优雅退出：终止时释放显存、关闭端口、清理子进程

## 基线（★ 必填）
- 基于 commit：`<git rev-parse --short HEAD>`
- 分支：`server/n53-2-graceful-exit`

## 改动
- 文件：`server/api.py`
- 摘要（≤5 行）：
  1. 捕获 SIGINT/SIGTERM，走 shutdown hook
  2. 停止接收新请求，等待在途请求完成（上限 5s）
  3. 显式释放模型与 CUDA 缓存
  4. 关闭监听端口
  5. 退出码 0（人为中断也返回 0，便于启动器区分"正常停止"与"崩溃"）

## 是否触碰接口
- [x] 否（接口字段 / 路径 / 错误码 / 响应语义均未变）

## 验收（必须有数字）
- 命令：起服务端 → `Ctrl+C` → `netstat -ano | findstr :8000` → `tasklist | findstr python`
- 数字（改前 → 改后）：残留监听 1 → 0；残留 python 进程 1 → 0；退出耗时 立即 → ≤5s
- 日志片段：`[shutdown] 在途请求 0，已释放显存，退出码 0`
- 补充：`回归.bat --suite mock` 8/8 PASS

## 遗留 / 风险 / 需要对方提供
- 客户端 `launcher.py` 会把退出码 0 视为正常停止，非 0 才提示异常
```

---

## 四、P3：E2xxx 错误码与统一文案

**分支**：`server/n53-4-err-codes`

```markdown
## 任务
N53-4 服务端业务失败统一为 E2xxx 错误码 + 给医生看的一句话

## 基线（★ 必填）
- 基于 commit：`<git rev-parse --short HEAD>`
- 分支：`server/n53-4-err-codes`

## 改动
- 文件：`server/api.py`、`docs/API_CONTRACT.md`
- 摘要（≤5 行）：
  1. 新增响应字段 `error_code`（形如 E2001），只加不改
  2. 现有 `message` 保持"给医生看的一句话"，与码一一对应
  3. 覆盖 4 类：未设目标 / 目标不存在 / 坏帧 / 内部异常
  4. 码表写入契约附录（E2xxx）
  5. 老客户端忽略新字段仍可正常工作（向后兼容验证）

## 是否触碰接口
- [x] 是 → 仅新增 `error_code` 字段；`message` 文案统一（语义不变）；**已通知闫**

## 验收（必须有数字）
- 命令：构造 4 类失败分别请求 A1/A2
- 数字（改前 → 改后）：可区分失败类型 0 → 4 类；每类码唯一
- 日志片段：4 条含 error_code 的响应样例
- 补充：`回归.bat --suite mock` 8/8 PASS；**去掉 error_code 解析的老客户端仍 8/8**

## 遗留 / 风险 / 需要对方提供
- 闫侧会把 E2xxx 映射到 GUI 提示与诊断包
```

---

## 五、P4：服务端可配置项清单

**分支**：`server/n53-3-config-list`

```markdown
## 任务
N53-3 服务端可配置项清单，并落地最常用的 3 项（端口 / 模型路径 / 并发上限）

## 基线（★ 必填）
- 基于 commit：`<git rev-parse --short HEAD>`
- 分支：`server/n53-3-config-list`

## 改动
- 文件：`server/api.py`、`docs/接口.md`、新增 `docs/服务端配置项.md`
- 摘要（≤5 行）：
  1. 列出全部可配置项与默认值、取值范围
  2. 落地 3 项：端口、YOLO/SAM2 权重根目录、并发上限
  3. 参数优先级：命令行 > 环境变量 > 默认值
  4. 与客户端 `config.toml` 的 `[server]` 段**互不覆盖**
  5. 非法值给出明确报错，不静默用默认

## 是否触碰接口
- [x] 否（启动参数变化，接口协议不变）

## 验收（必须有数字）
- 命令：改端口为 8010 启动 → `curl http://127.0.0.1:8010/healthz`
- 数字（改前 → 改后）：可配置项 0 → 3 项生效；非法端口报错信息 1 条
- 日志片段：`[config] port=8010 weights=mixed source=cli`
- 补充：`回归.bat --suite mock` 8/8 PASS

## 遗留 / 风险 / 需要对方提供
- 闫把这份清单并入客户端配置文档，避免两处口径不一致
```

---

## 六、P5：长跑守护与观测（对应 N46 + N53-5）

**分支**：`server/n46-soak-guard`

```markdown
## 任务
N46 服务端长跑健壮性：进程守护/自恢复 + 观测指标留档（支撑 N53-5 长稳 1h）

## 基线（★ 必填）
- 基于 commit：`<git rev-parse --short HEAD>`
- 分支：`server/n46-soak-guard`

## 改动
- 文件：`server/*.py`、新增 `scripts/soak_report.py`
- 摘要（≤5 行）：
  1. 异常退出后自恢复（或明确失败并留证据，不静默重试）
  2. 每 60s 记录：显存占用、句柄数、A1 P95 漂移
  3. 输出 `runs/soak/<时间戳>/soak.csv`
  4. 达到阈值告警（显存 >90%、P95 漂移 >30%）
  5. 与客户端 `run_soak.py` 共用时间戳，便于对齐

## 是否触碰接口
- [x] 否

## 验收（必须有数字）
- 命令：`回归.bat` → 长跑 2h（N46 标准）
- 数字（改前 → 改后）：无人工干预时长 0h → 2h；显存增长 <5%；P95 漂移 <30%
- 日志片段：soak.csv 首尾两行 + 告警（若有）
- 补充：结束后 `netstat`/`tasklist` 无残留

## 遗留 / 风险 / 需要对方提供
- 闫侧的长稳只跑 1h（N53-5）；2h 数据以服务端为准
```

---

## 七、发给李的话（可直接复制到桌面/群里）

> 老李，软件化这块我拆成 5 个 PR 派给你，都在 `main` 上开短分支，单个 ≤2 天：
>
> - **P1 `server/n53-1-healthz`**：最高优先，`/healthz` 加 `version` + `ready` + `warmup_ms`。我这边启动器要靠 `ready` 判断"能不能开界面"，现在只能猜日志。
> - **P2 `server/n53-2-graceful-exit`**：Ctrl+C 之后端口和进程要干净，退出码 0 当正常停止。
> - **P3 `server/n53-4-err-codes`**：A1/A2 失败给 `error_code`（E2xxx），`message` 统一成给医生看的一句话，只加字段。
> - **P4 `server/n53-3-config-list`**：端口/权重路径/并发上限这 3 项做成可配，再给一份清单，我并进 `config.toml`。
> - **P5 `server/n46-soak-guard`**：2h 长跑守护 + 显存/句柄/P95 漂移留档。
>
> 顺序按 P1 → P2 → P3（都要动 `api.py`，串行开分支，别并行改）；P4、P5 随时。
> 每个 PR 用仓库模板填，**「基于 commit」那栏开工时 `git rev-parse --short HEAD` 填上**，验收栏必须有改前→改后的数字，CI 绿了再喊我 review。
> 接口只加字段（走 v1.0.4），P1 和 P3 合之前跟我说一声，我同步契约文档和客户端解析。

---

## 八、李侧的开工步骤（照做即可）

```powershell
cd D:\EgoMed-Agent
git pull --rebase
git status                     # 应干净
git rev-parse --short HEAD     # ★ 记下，PR 里要填

git switch -c server/n53-1-healthz
# 改 server/api.py + docs
python -m py_compile server/api.py
回归.bat --suite mock           # 应 8/8

git add server/api.py docs/API_CONTRACT.md docs/接口.md   # 不要 git add -A
git commit -m "feat(server): N53-1 /healthz 增 version 与就绪标志"
git -c http.proxy=http://127.0.0.1:1080 -c https.proxy=http://127.0.0.1:1080 push -u origin server/n53-1-healthz
# 然后在 GitHub 新建 PR，base 选 main，粘贴第二节的描述
```

---

## 九、四个容易被门禁挡回来的点

1. **「基于 commit」空着** → 模板第一项就是它，防止底本过期；
2. **`git add -A`** → 会带上 `runs/`、临时文件；只 `git add` 改动文件；
3. **验收只写"已通过"没数字** → 一律退回，要求改前→改后；
4. **PR 生命周期超 2 天** → 冲突概率陡增，拆小或先合一半。

---

## 十、与闫侧（N53-1~4）的对接点

| 李的 PR | 对应闫的增量 | 对接形式 |
|---|---|---|
| P1 | N53-1 统一入口 | 闫的 `demo` 轮询 `/healthz.ready` |
| P2 | N53-1 统一入口 | 闫按退出码判断"正常停止/异常退出" |
| P3 | N53-4 错误码与诊断 | 闫把 E2xxx 映射到 GUI 提示与诊断包 |
| P4 | N53-3 配置持久化 | 闫并入 `config.toml` 的 `[server]` 段 |
| P5 | N53-5 长稳 1h | 两边共用时间戳，报告合并 |
