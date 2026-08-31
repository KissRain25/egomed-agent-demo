# EgoMed-Agent 项目进度记录

> 最后更新：2026-08-29
> 本文档记录项目从 0 到当前的全部进度，供团队成员随时查阅。

---

## 一、项目最终目标

**智能眼镜 AI 医学影像辅助系统**
- 使用者：医生
- 形态：戴智能眼镜（第一人称视角）→ 与系统对话 → 实时识别屏幕上的医学影像（MRI/CT/超声/X光/内窥镜）→ 精确分割器官/病变区域 → 跨帧跟踪

## 二、整体架构

```
医生（戴智能眼镜）→ 语音指令
        ↓
[语音识别] 声音 → 文字（ASR）        ← 待开发（下一步，需权重）
        ↓
[自然语言理解] 文字 → 分割目标（NLU） ← 完成（新增）
        ↓
[第一人称视频/屏幕画面] ← 眼镜摄像头实时采集  ← 待开发
        ↓
[模态分类] 认出是 MRI/CT/超声/X光/内窥镜     ← 完成（分类器权重已下载）
        ↓
[目标检测 YOLO] 在画面里找到目标            ← 完成（5 模态权重全部就位）
        ↓
[分割 SAM2] 精确分割出器官/病变区域         ← 完成（5 模态全部支持）
        ↓
[结果显示/语音反馈] 展示给医生              ← 完成（Gradio 界面）
```

## 三、当前进度

### ✅ 已完成

| # | 事项 | 详情 | 完成日期 |
|---|------|------|---------|
| 1 | 数据下载 | 5 模态 697 个第一人称视频全齐 | 2026-08-28 |
| 2 | 环境确认 | 现成 `egomed` 环境（Python 3.12.13 + torch 2.8.0+cu128 + GPU RTX 5060） | 2026-08-28 |
| 3 | demo 跑通 | CLI 模式成功分割 60 帧 LV cavity | 2026-08-28 |
| 4 | Gradio 网页界面 | `http://127.0.0.1:7860` 已启动 | 2026-08-28 |
| 5 | 视频编码修复 | OpenCV mp4v → H.264，浏览器可播放 | 2026-08-28 |
| 6 | 自然语言识别（NLU） | "帮我分割左心室"→LV cavity，中英文/否定/歧义/纠错，`scripts/demo/nl_parser.py` | 2026-08-28 |
| 7 | 下载 5 模态权重 | 从 hf-mirror 的 EgoMed-IEMIS models 下载 6 个权重（免训练直接推理），`scripts/download_models.py` | 2026-08-29 |
| 8 | 多模态分割接入 | 引擎启用 5 模态 + demo 支持模态自动匹配 + NLU 扩展全器官词典（23/23 自测通过） | 2026-08-29 |
| 9 | **4 模态数据准备** | 从各模态 `rawdata` 视频提取 img 帧：Amos 273 病例 / CAMUS 80 / Montgomery 150 / PolypGen 97，`scripts/prepare_rawdata_frames.py` | 2026-08-29 |
| 10 | **4 模态分割验证通过** | Amos(肝)/CAMUS(左室)/Montgomery(左肺)/PolypGen(息肉) 全部检测+分割成功 | 2026-08-29 |
| 11 | demo 支持 `--modality` | CLI 可显式指定模态，解决跨模态歧义（如左心室在 ACDC/CAMUS 都有） | 2026-08-29 |

### ⏭️ 待办（按优先级）

| # | 事项 | 说明 | 负责人建议 |
|---|------|------|-----------|
| 1 | 语音识别（ASR） | 语音转文字（NLU 已就绪，接上 ASR 即可真·对话） | 队友 |
| 2 | 智能眼镜实时采集 | 接入第一人称视频流（当前是选文件） | 团队交叉 |
| 3 | 数据格式规范对接 | 见 `DATA_FORMAT_AGREEMENT.md` | 你 |

## 四、关键技术信息

### 环境
```
conda 环境名: egomed（位于 D:\ScienceApp\anaconda\envs\egomed）
Python: 3.12.13
torch: 2.8.0+cu128
GPU: NVIDIA RTX 5060 (8GB, CUDA 13.1)
sam2: 已装
ultralytics: 8.4.6
gradio: 6.26.0
imageio-ffmpeg: 0.6.0（用于 H.264 转码，已装）
```

### 权重文件
| 权重 | 路径 | 作用 |
|------|------|------|
| SAM2 | `D:\EgoMed-Agent\checkpoints\sam2.1_hiera_base_plus.pt` | 分割模型 |
| 模态分类器 | `D:\EgoMed-Agent\runs\tool_selection_cls\tool_selection_yolo26m_cls_imgsz320\weights\best.pt` | 识别影像模态 |
| MRI 检测器 | `D:\EgoMed-Agent\runs\yolo26_det\ACDC_yolo26m_imgsz1024\weights\best.pt` | MRI 心脏目标检测 |
| CT 检测器 | `D:\EgoMed-Agent\runs\yolo26_det\Amos_yolo26m_imgsz1024\weights\best.pt` | CT 腹部目标检测 |
| 超声检测器 | `D:\EgoMed-Agent\runs\yolo26_det\CAMUS_yolo26m_imgsz1024\weights\best.pt` | 心脏超声目标检测 |
| X光检测器 | `D:\EgoMed-Agent\runs\yolo26_det\Montgomery-County-CXR-Set_yolo26m_imgsz1024\weights\best.pt` | 胸片目标检测 |
| 内窥镜检测器 | `D:\EgoMed-Agent\runs\yolo26_det\PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024\weights\best.pt` | 内窥镜息肉检测 |

> 5 个模态的检测器权重均下载自 `hf-mirror.com/datasets/daizywang/EgoMed-IEMIS`，
> 是作者在 EgoMed 数据上训练好的 YOLO26m 权重，**免训练、直接推理**。

### 数据位置
```
D:\BaiduNetdiskDownload\EgoMed-Screen\
├── ACDC/                          (97 个视频, MRI 心脏)
├── Amos/                          (273 个视频, 腹部 CT)
├── CAMUS/                         (80 个视频, 心脏超声)
├── Montgomery-County-CXR-Set/     (150 个视频, 胸片 X 光)
└── PolypGen2021_MultiCenterData_v3/  (97 个视频, 内窥镜)
```

### demo 测试数据
```
D:\EgoMed-Agent\data\ACDC\img\<病例号>   （1~72，每个约120帧心脏MRI）
已验证: 病例14 → LV cavity 分割成功
```

### 运行命令
```bash
# 启动 Gradio 网页界面（目标输入支持自然语言）
D:\ScienceApp\anaconda\envs\egomed\python.exe scripts\demo\egomed_demo.py

# CLI 一次性分割（--target 支持自然语言）
D:\ScienceApp\anaconda\envs\egomed\python.exe scripts\demo\egomed_demo.py \
    --cli --source D:\EgoMed-Agent\data\ACDC\img\14 --target "帮我分割左心室"
```

### 自然语言解析模块（NLU）
```
- 位置：`scripts/demo/nl_parser.py`（零依赖，纯规则 + 词典，离线微秒级）
- 能力：中英文同义词、否定排除、歧义提示、英文拼写纠错、多模态自动匹配
- 支持的器官（全 5 模态）：
    · MRI心脏: 左心室腔 / 右心室腔 / 心肌
    · 超声心脏: 左室心肌 / 左心室 / 左心房
    · CT腹部: 脾 / 肾 / 肝 / 胃 / 胆囊 / 食道 / 胰腺 / 肾上腺 / 十二指肠 / 膀胱 / 主动脉 / 下腔静脉 / 前列腺 / 子宫
    · X光胸片: 左肺 / 右肺
    · 内窥镜: 息肉
- 扩展：以后接语音识别（ASR）时，只需"语音→文字"再调用 parse_target() 即可复用
```

### demo 输出位置
```
D:\EgoMed-Agent\runs\demo_output\<时间戳>\
├── result.mp4      # H.264 分割动画（浏览器可播）
├── masks\          # 逐帧分割掩码
└── overlay\        # 逐帧效果图（红色=预测mask, 蓝框=YOLO检测, 黄框=SAM2追踪）
```

## 五、已验证的分割效果

- **输入**：`data/ACDC/img/14`（心脏 MRI 帧序列）
- **目标**：LV cavity（左心室）
- **结果**：60 帧全部检测 + 分割成功
- **输出**：`runs\demo_output\20260828_182503\result_h264.mp4`

## 六、团队分工

| 人 | 职责 |
|----|------|
| 小李（你） | 数据 + 视频整理（EgoMed-Screen） |
| 小王（队友） | 代码 + 跑通流程（EgoMed-Agent） |
| 交叉 | 模型训练/验证 |

> 数据格式约定见 `DATA_FORMAT_AGREEMENT.md`

## 七、注意事项

1. **第一人称 MP4 不能直接测分割**：`rawdata/` 里的 MP4 是录屏画面，YOLO 检测不到器官。要用纯影像帧文件夹测（`data/<模态>/img/<病例号>`）。**注意**：现在所有 5 模态的 img 帧都已备好（ACDC 原已有，Amos/CAMUS/Montgomery/PolypGen 已用 `scripts/prepare_rawdata_frames.py` 提取），但跨模态歧义目标（如左心室）需显式指定模态（CLI 加 `--modality`，GUI 用下拉框）。
2. **系统默认 Python 是 3.14**，跑项目必须用 `egomed` 环境的 python（3.12）。
3. **Gradio 服务重启**：改代码后需重启进程（停掉旧的 python.exe，重新跑启动命令）。
4. 启动脚本：`D:\EgoMed-Agent\start_demo.py`（后台启动 Gradio，日志在 `runs\gradio_log.txt`）。
5. **系统代理会干扰 Gradio 启动**：`egomed_demo.py` 已内置禁用本地代理逻辑，无需手动设置。
