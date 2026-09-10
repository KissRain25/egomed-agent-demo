# EgoMed-Agent 接口约定（API Contract）

> 版本：v0.2（**字段已定稿，待联调通过后冻结为 v1.0**）
> 创建日期：2026-09-10 ｜ 更新日期：2026-09-10
> 参与人：闫、李（如有第三人请补充）
> 用途：定义「眼镜端（客户端）」与「服务器端」之间所有通信的消息格式。
> 规则：**接口一旦冻结，两端独立开发；任何字段变更必须同步对方并更新本文档。**
> 说明：v0.1 中所有"待定"项已在 v0.2 拍板定死，逐条依据见第八节《决议记录》。**本节以下内容即为开发依据，不需要再等谁确认。**

---

## 一、通信总览（4 条链路）

| # | 链路 | 方向 | 传什么 | 数据量 | 实时性 | 计划接口 |
|---|------|------|--------|--------|--------|----------|
| 1 | 画面上行 | 眼镜 → 服务器 | 一帧医学影像画面 | 大 | 高 | `POST /api/v1/frame` |
| 2 | 语音上行 | 眼镜 → 服务器 | 医生说话的短录音 | 小 | 中 | `POST /api/v1/audio` |
| 3 | 状态查询 | 眼镜 → 服务器 | 查当前会话状态 | 极小 | 低 | `GET /api/v1/state` |
| 4 | 结果下行 | 服务器 → 眼镜 | 分割结果 + 提示消息 | 大/小 | 高 | 先用 frame 的响应返回，后期升级 WebSocket |

**已定策略（v0.2）**：

1. 本期**只做链路 1 和链路 2**（`A1` / `A2`）——这两条已经构成完整闭环：医生说话 → 指定目标 → 服务器分割 → 结果回显。
2. 链路 3（状态查询）、链路 4（WebSocket）**本期暂缓**。理由：A1 的响应里已经带回了模态、目标、提示消息，客户端自己存一份状态即可，省两个接口。
3. 全部走 **HTTP 一问一答**，后期若带宽或延迟不够，再把"结果下行"升级为 WebSocket（升级不改变现有字段）。

---

## 二、通用约定（两端都必须遵守）

| 项 | 约定 |
|----|------|
| `session_id` | 客户端首次连接时生成，格式 `doc-<医生编号>-<yyyymmdd>-<4位随机>`，如 `doc-001-20260910-a3f7`。**一次医生使用过程一个**（不是一帧一个，也不是一个病人一个） |
| 时间戳 | Unix 毫秒整数，字段名统一 `timestamp` |
| 请求图片 | **multipart/form-data 上传 jpg**；规格：长边 ≤1280 px、质量 80、单帧 ≤300 KB（客户端负责压缩） |
| 响应图片 | **base64 字符串内嵌 JSON**，字段 `overlay`；规格同上（长边 ≤1280、质量 80） |
| 字符编码 | 一律 UTF-8 |
| 接口版本 | 统一前缀 `/api/v1` |
| 错误格式 | `{"status": "error", "code": 1001, "message": "人话描述"}` |
| HTTP 状态码 | 200 成功（**业务上的"需要澄清"也算 200**，用字段表达）；400 参数错；404 无此会话；500 服务器内部错误 |

---

## 三、接口清单（v0.2 已定）

| 编号 | 接口 | 用途 | 优先级 | 本期做不做 | 主责 |
|------|------|------|--------|-----------|------|
| A1 | `POST /api/v1/frame` | 上传一帧画面，返回分割结果 | P0 | **做** | 李 |
| A2 | `POST /api/v1/audio` | 上传录音，返回识别文本 + 分割目标 | P0 | **做** | 李 |
| A3 | `GET /api/v1/state` | 查询当前会话状态 | P3 | 暂缓 | 李 |
| A4 | `WS /api/v1/ws/{session_id}` | 服务器主动推结果 | P3 | 暂缓 | 李 |

---

## 四、接口详情（v0.2 定稿）

### A1. POST /api/v1/frame —— 上传一帧画面

**请求**（multipart/form-data）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 会话标识 |
| `timestamp` | int | 是 | 采集时间（Unix ms） |
| `image` | file(jpg) | 是 | 一帧画面，长边 ≤1280、≤300 KB |

**响应**（application/json）：

```json
{
  "status": "ok",
  "session_id": "doc-001-20260910-a3f7",
  "timestamp": 1789000000123,
  "modality": "ACDC (MRI 心脏)",
  "target": "LV cavity",
  "overlay": "<base64 jpg：带分割标注的结果图>",
  "message": "已锁定左心室",
  "need_confirm": false,
  "candidates": [],
  "elapsed_ms": 187,
  "frames_processed": 152
}
```

**字段说明（已定，无待定项）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `ok` / `error` |
| `session_id` | string | 原样回传 |
| `timestamp` | int | 服务器处理时间（Unix ms） |
| `modality` | string | 识别出的模态；未识别时为 `""`（**不报错**，见下方规则） |
| `target` | string | 当前跟踪目标；无目标时为 `""` |
| `overlay` | string | base64 jpg（无结果时为空字符串 `""`） |
| `message` | string | 给医生看的一句话提示，客户端可直接显示或语音播报 |
| `need_confirm` | bool | 需要医生澄清时为 `true` |
| `candidates` | array | `need_confirm=true` 时给出候选目标，如 `["左心室", "右心室"]`；否则为 `[]` |
| `elapsed_ms` | int | 本帧处理耗时（客户端可显示，用于性能观察） |
| `frames_processed` | int | 该 session 累计处理帧数 |

**两个边界情况的处理规则（已定）**：

1. **画面不可用 / 未识别出模态**（模糊、过暗、没对准屏幕、不是医学影像）
   → 返回 `status:"ok"`，`modality:""`、`target:""`、`overlay:""`，`message:"没看清屏幕上的影像，请对准屏幕再试一次"`。
   **理由**：这是日常常态，不是异常，不该让客户端走错误分支。

2. **跨模态歧义**（如"左心室"在 MRI 和超声里都存在）
   → 返回 `need_confirm:true`，`candidates:["ACDC (MRI 心脏)","CAMUS (超声)"]`，`message:"左心室在 MRI 和超声里都有，请问是哪一个？"`。
   客户端收到 `need_confirm=true` 时：显示 `message`，等医生再说一句话 → 走 A2。

**客户端行为约定**：拿到 `overlay` 非空就显示；`overlay` 为空时保留上一帧画面并显示 `message`，不要黑屏。

**错误响应示例**：

```json
{ "status": "error", "code": 2001, "message": "画面不可用，请对准屏幕" }
```

### A2. POST /api/v1/audio —— 上传语音

**请求**（multipart/form-data）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 会话标识 |
| `timestamp` | int | 是 | 录音结束时间（Unix ms） |
| `audio` | file(wav) | 是 | 16 kHz、单声道、**≤5 秒**、≤200 KB |

**响应**：

```json
{
  "status": "ok",
  "session_id": "doc-001-20260910-a3f7",
  "text": "帮我分割左心室",
  "target": "LV cavity",
  "modality_hint": "ACDC (MRI 心脏)",
  "need_confirm": false,
  "candidates": [],
  "message": "已切换到左心室"
}
```

**字段说明**：`text` 为 ASR 识别原文（便于排查识别错误）；`target` 为 NLU 解析出的分割目标；`modality_hint` 为建议模态，可为 `""`；其余同 A1。

**客户端行为约定**：采用 **PTT（按住说话，松开上传）**；录音期间界面显示"正在听…"，上传后显示"识别中…"。

### A3. GET /api/v1/state?session_id=xxx —— 本期暂缓

保留设计，不实现：

```json
{
  "status": "ok",
  "session_id": "doc-001-20260910-a3f7",
  "modality": "ACDC (MRI 心脏)",
  "target": "LV cavity",
  "frames_processed": 152,
  "last_update": 1789000000000
}
```

### A4. WebSocket /api/v1/ws/{session_id} —— 本期暂缓

- 服务器在每次处理完一帧后主动推送结果，避免客户端轮询
- 消息格式与 A1 响应一致

---

## 五、状态与生命周期语义（v0.2 已定，最容易返工的部分）

### 5.1 会话生命周期

| 事项 | 约定 |
|------|------|
| 谁生成 | 客户端生成，首次发请求时带上 |
| 何时新建 | 医生开始使用系统时；同一次使用过程中**不换** session_id |
| 服务器保存什么 | 当前模态、当前目标、帧计数、最近一次处理时间 |
| 何时清理 | 超过 **30 分钟**无请求，服务器可丢弃该 session 状态 |
| 如何重置 | 客户端直接生成一个新 session_id 即可（本期不做 reset 接口） |

### 5.2 目标切换规则

- **只有 A2 语音显式指令才能切换目标**（如"换成右心室"）
- 画面上出现新目标、或模型检测到别的器官，**不自动切换**
- 切换后：服务器重新执行「检测 + 首次分割」，客户端清空旧 overlay
- 切换之后：后续帧只做**跟踪**（IOU re-track），不再重复检测
- **理由**：避免目标自己乱跳，医生始终掌握控制权

### 5.3 模态切换防抖

- 服务器每帧都做模态分类，但**连续 3 帧分类结果一致才正式切换**（阈值可配 `MODALITY_SWITCH_N = 3`）
- 单帧分类置信度低于阈值（建议 0.5）时，**保持原模态不变**
- 模态切换时清空当前 `target`，并提示：`"已切换到 CT，请告诉我要分割哪个器官"`

### 5.4 帧率与丢帧

| 事项 | 约定 |
|------|------|
| 客户端发送帧率 | 默认 **5 fps**（每 200 ms 一帧），可配 |
| 上一帧还没返回 | **不发新帧**（避免堆积） |
| 服务器忙不过来 | **丢旧帧，不排队**：单会话串行处理，队列长度 = 1，新帧覆盖未处理的旧帧 |
| 后续优化（本期不做） | 画面无变化时不发帧（医学屏幕几乎静止，能省大量流量） |

### 5.5 超时与重试

| 事项 | 约定 |
|------|------|
| 单请求超时 | **3 秒**（可配） |
| 超时/失败后 | **不自动重试**（避免雪崩），客户端显示提示，继续发下一帧（相当于自然重试） |
| 连续失败 3 次 | 提示"连接异常，请检查服务器"，并暂停发送 |
| 恢复 | 客户端每 5 秒探测一次，成功后自动恢复 |

### 5.6 性能预算（服务器侧目标）

| 场景 | 目标耗时 |
|------|---------|
| 首次点名分割（分类 + YOLO + SAM2 首帧） | ≤ 1.5 s |
| 后续跟踪帧 | ≤ 200 ms |
| 日志 | 服务器**每帧记一行**：`session / 帧号 / 模态 / 目标 / 耗时 / 状态`，联调排查全靠它 |

---

## 六、错误码表（v0.2 已定）

| code | 含义 | 客户端应该做什么 |
|------|------|------------------|
| 1001 | 参数缺失或格式错误 | 检查客户端代码（多半是 bug） |
| 1002 | session 不存在或已过期 | 重新生成 session_id |
| 2001 | 画面不可用（模糊 / 过暗 / 无有效内容） | 提示"请对准屏幕"，继续发帧 |
| 2002 | 未识别出模态 | 提示"没认出这是什么影像"，继续发帧 |
| 2003 | 目标无法确定（需澄清） | 配合 `need_confirm`，提示医生补充说明 |
| 5000 | 服务器内部错误 | 提示"服务器异常"，记录日志，继续发帧 |

> 统一格式：`{"status": "error", "code": 2001, "message": "画面不可用，请对准屏幕"}`

---

## 七、分工与文件归属（避免两人改同一文件）

| 工作项 | 主责 | 配合 | 产出 | 预估 |
|--------|------|------|------|------|
| 列出所有通信场景 + 定字段 | **闫（主持）** | 李 | 会议结论 | 1h |
| 撰写/维护 `API_CONTRACT.md` | **闫** | 李 review（只提意见，不直接改） | 本文档 | 2h |
| 服务端接口骨架（假实现，先返回固定图） | **李** | 闫提供测试样例 | `server/api.py` 可被调用 | 0.5 天 |
| 服务端接入真引擎 | **李** | — | 真分割结果 | 后续 |
| 客户端调用脚本 / 模拟页面 | **闫** | 李接口答疑 | `client/mock_client.py` 或网页 | 0.5 天 |
| 联调 + 验收 | **两人一起** | — | 端到端跑通 | 0.5 天 |
| 评审 | 导师 / 师兄 | — | 反馈意见 | — |

**文件归属（硬约定）**

| 路径 | 归属 | 说明 |
|------|------|------|
| `API_CONTRACT.md` | 闫 | 李只提意见，避免并发编辑冲突 |
| `server/` | 李 | 服务端全部代码 |
| `client/mock_client.py`、模拟网页 | 闫 | 客户端/测试工具 |
| `scripts/`（现有引擎） | 不动 | 只 import 复用，不修改 |

---

## 八、决议记录（v0.2 已拍板）

> 本节是 v0.1 中所有"待定"项的最终答案。**双方按此开发，不需要再开会讨论。**

| # | 议题 | 结论 | 依据 / 理由 |
|---|------|------|-------------|
| 1 | 接口范围 | 只做 A1、A2；A3、A4 暂缓 | 两条已构成完整闭环，少写两个接口 |
| 2 | 画面怎么传 | **multipart 文件上传 jpg** | 比 base64 省约 1/3 体积，两端都原生支持 |
| 3 | 结果怎么回 | **回整张 overlay 图**（base64 内嵌 JSON） | 客户端零叠加逻辑；带宽紧张再改回 mask |
| 4 | 通信方式 | **纯 HTTP 一问一答** | 最好调试；WebSocket 留到后期 |
| 5 | 服务器"反问"机制 | `need_confirm=true` + `message` + `candidates` | 复用已有字段，客户端只需弹提示 |
| 6 | `session_id` | 客户端生成，一次医生使用一个 | 语义清晰，服务器无需管理会话创建 |
| 7 | 目标切换 | **仅语音显式指令可切换** | 防止目标乱跳，医生掌控 |
| 8 | 模态切换 | 连续 3 帧一致才切，低置信保持原模态 | 防抖动，避免画面闪烁 |
| 9 | 帧率与丢帧 | 5 fps；上一帧未返回不发新帧；服务器丢旧帧不排队 | 保证低延迟，不积压 |
| 10 | 超时与重试 | 3 s 超时，不自动重试，连续失败 3 次暂停 | 避免雪崩 |
| 11 | 错误码 | 6 个：1001 / 1002 / 2001 / 2002 / 2003 / 5000 | 覆盖已知全部异常场景 |
| 12 | 接口冻结时间 | 端到端联调通过当天冻结为 v1.0 | 冻结后变更走四步流程 |

**待导师 / 硬件确认的 4 项（已给暂定假设，不阻塞开发）**：

| 事项 | 暂定假设（开发按这个走） | 影响 |
|------|--------------------------|------|
| 眼镜型号与接口 | 假设能拿到"一帧图像"（UVC 摄像头或 RTSP 流）；开发期用笔记本摄像头 / 手机模拟 | 只影响采集适配层，不影响接口 |
| 服务器放哪 | 假设本地 RTX 5060，局域网访问 | 影响地址配置，不影响字段 |
| 回显介质 | 假设手机 / 副屏显示；AR 镜片作为可选增强 | 影响客户端形态，不影响接口 |
| 数据合规 | 假设只用公开数据集（EgoMed-Screen 等），不接触真实患者数据 | 若导师要求接触院内数据，需重新评估 |

---

## 附录 A：服务端实现前置信息（给李的 agent 直接照做）

> 本节是"能立刻开工"所需的技术前提，不是接口约定本身。**照着做即可，不要自由发挥。**

### A.1 运行环境（必须用这个解释器）

```
D:\ScienceApp\anaconda\envs\egomed\python.exe      ← Python 3.12.13 + torch 2.8.0+cu128 + RTX 5060
```

**系统默认 Python 是 3.14，装不了这些依赖，绝对不能用。**

### A.2 依赖现状（已实测）

| 包 | 状态 | 说明 |
|----|------|------|
| fastapi / uvicorn | 已装 | 直接用 |
| requests | 已装 | 客户端用 |
| faster-whisper | 已装 | ASR 用 |
| **python-multipart** | **未装** | FastAPI 收 multipart 必需，**不装会直接报错/500，第一步先装** |

安装命令（国内源）：

```bash
D:\ScienceApp\anaconda\envs\egomed\python.exe -m pip install python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### A.3 端口与目录约定

| 项 | 约定 |
|----|------|
| 端口 | **8000**（本地联调）；需要局域网访问时监听 `0.0.0.0`，否则 `127.0.0.1` |
| 新增代码 | 只放在 `D:\EgoMed-Agent\server\` 下，入口 `server/api.py` |
| 流式临时目录 | `D:\EgoMed-Agent\runs\stream\<session_id>\frames\`（滚动帧目录） |
| 日志 | `D:\EgoMed-Agent\runs\server_log.txt`，每帧一行 |

### A.4 可复用的现成代码（**只 import，不修改**）

`scripts/demo/egomed_demo.py` 已经把引擎加载好了，直接这样用：

```python
import sys
sys.path.insert(0, r"D:\EgoMed-Agent\scripts\demo")
import egomed_demo as demo          # 内部已加载引擎模块 eng，并配好 CUDA_VISIBLE_DEVICES

from nl_parser import parse_target        # 文字 -> 分割目标
from asr_engine import transcribe         # 语音 -> 文字
```

| 你要的功能 | 现成入口 | 返回 |
|---|---|---|
| 目标 → 模态 | `demo.resolve_modality(target_text, modality=None)` | `(modality_name, name_to_gray)` |
| 加载模型（**带缓存，进程内只加载一次**） | `demo._get_models(modality)` | `(sam2_predictor, yolo_model)` |
| 视频/文件夹 → 帧列表 | `demo.resolve_frames(source_path, work_dir, max_frames)` | `(image_files, frame_dir)` |
| 检测+分割+跟踪主循环 | `demo._segment_target(...)` | `dict`（含 `overlay_files`、`masks`、`status`） |
| 语音 → 文字 | `asr_engine.transcribe(audio_path)` | `str` |
| 文字 → 目标 | `nl_parser.parse_target(text)` | `ParseResult`（`.targets`、`.ok`、`.message`） |
| 引擎本体（一般不用直接碰） | `demo.eng` | 提供 `init_sam2_predictor`、`YOLO`、`precompute_yolo_detections`、`DET_CONF_THRES` |

**支持的 target 名称（必须用这些原词，别自己翻译）**：

| 模态键名 | 可选目标 |
|---|---|
| `ACDC (MRI 心脏)` | `LV cavity` / `RV cavity` / `myocardium` |
| `CAMUS (超声 心脏)` | `Left Ventricle` / `Left Ventricular Myocardium` / `Left Atrium` |
| `Amos (CT 腹部)` | `liver` / `spleen` / `left kidney` / `right kidney` / `stomach` / `gall bladder` / `pancreas` / `arota` / `postcava` / `esophagus` / `bladder` / `duodenum` / `left adrenal gland` / `right adrenal gland` / `prostate/uterus` |
| `Montgomery (X光 胸片)` | `left lung` / `right lung` |
| `PolypGen (内窥镜)` | `polyp` |

### A.5 关键：现有引擎是"离线批处理"，必须做在线化改造

现状：`_segment_target` 会 `predictor.init_state(video_path=frame_dir)` —— **把整个帧目录一次性灌进 SAM2**，并 `precompute_yolo_detections` 预计算所有帧。它按"一次跑完一段视频"设计，不能直接按单帧调用。

**P0 采用滚动窗口（伪流式）方案，不要重写引擎**：

```
每收到一帧：
  1. 写入 runs\stream\<session_id>\frames\，文件名按时间递增（如 000123.jpg）
  2. 只保留最近 N 帧（建议 N=10），更早的删掉
  3. 对该目录跑一次 demo._segment_target(...)
  4. 只把"最新一帧"的 overlay 转成 base64 返回
```

- 优点：100% 复用已验证代码，当天可跑通
- 代价：重复计算（窗口 10 帧 ≈ 每帧 50ms 检测 × 10）；医学屏幕画面几乎静止，可接受
- 后续优化（P2 再做）：用 SAM2 流式 API（`add_new_points` / `add_new_mask`，手动维护 inference_state）做真流式
- **P0 阶段明确不做的事**：不重写引擎、不修改 `scripts/` 下任何文件

### A.6 T4 假实现要求（先做这个，让闫当天能联调）

`/api/v1/frame` 先返回**固定结果**即可：

- `overlay`：直接把请求里的图原样 base64 回去，或返回一张纯色图
- `message`：`"假实现：已收到画面"`
- `modality` / `target`：给个固定值，如 `"ACDC (MRI 心脏)"` / `"LV cavity"`
- **字段结构必须与第四节 A1 完全一致**（一个都不能少、名字不能改）

`/api/v1/audio` 假实现同理：`text` 固定返回 `"帮我分割左心室"`，`target` 返回 `"LV cavity"`。

### A.7 测试样例（已存在，可直接用）

| 用途 | 路径 |
|---|---|
| 心脏 MRI（已验证 LV cavity） | `D:\EgoMed-Agent\data\ACDC\img\14\14_0000.jpg`（共 120 帧，`14_0000.jpg` ~ `14_0119.jpg`） |
| CT 腹部 | `D:\EgoMed-Agent\data\Amos\img\<病例号>\` |
| 心脏超声 | `D:\EgoMed-Agent\data\CAMUS\img\<病例号>\` |
| 胸片 X 光 | `D:\EgoMed-Agent\data\Montgomery-County-CXR-Set\img\<病例号>\` |
| 内窥镜 | `D:\EgoMed-Agent\data\PolypGen2021_MultiCenterData_v3\img\<病例号>\` |

### A.8 启动与自检命令（可直接复制）

```bash
# 启动服务
D:\ScienceApp\anaconda\envs\egomed\python.exe D:\EgoMed-Agent\server\api.py

# 验证 A1（multipart 上传一帧）
curl.exe -F "session_id=test-001" -F "timestamp=1789000000000" -F "image=@D:\EgoMed-Agent\data\ACDC\img\14\14_0000.jpg" http://127.0.0.1:8000/api/v1/frame

# 验证 A2（multipart 上传音频）
curl.exe -F "session_id=test-001" -F "timestamp=1789000000000" -F "audio=@test.wav" http://127.0.0.1:8000/api/v1/audio
```

### A.9 禁止事项（违反会导致返工）

1. 不修改 `scripts/` 下的任何现有文件（引擎、demo、ASR、NLU 都是已验证代码，只 import）
2. 不修改本文档与 `接口.md`（有意见提给闫，由闫统一改）
3. 不用系统 Python 3.14，不用新环境
4. 不把模型加载写进请求处理函数里（必须进程启动时加载一次并常驻）
5. 不擅自增删或改名响应字段

---

## 九、变更记录

| 日期 | 版本 | 变更内容 | 人 |
|------|------|----------|-----|
| 2026-09-10 | v0.1 | 创建草案 | 闫 |
| 2026-09-10 | v0.2 | 全部待定项拍板定死：接口范围收敛为 A1/A2、确定 multipart 上传与 overlay 返回、新增状态与生命周期语义（目标切换 / 模态防抖 / 帧率丢帧 / 超时重试 / 性能预算）、新增 6 个错误码、新增决议记录；新增附录 A（服务端实现前置信息：运行环境 / 依赖缺口 / 复用入口 / 在线化改造方案 / 测试样例 / 启动自检命令） | 闫 |
