# 交付形态说明 —— 从 bat 到 exe（2026-11-04，v0.2.4）

> 一句话：**`egomed.bat` 只是"找 python.exe + 转发"的入口壳，程序本体早就写好了**
> （`launcher.py` + `client/live_console.py`，tkinter 控制台，20 个子命令）。
> 这次做的是把**入口壳换成真正的可执行程序**，让**目标机器不装 Python 也能双击打开界面**。

## 一、现在有几种启动方式

| 方式 | 目标机器需要装 Python 吗 | 用途 | 状态 |
|---|---|---|---|
| `egomed.bat` | **需要**（且要 egomed 环境） | 开发者日常、脚本化 | 一直在用，保留 |
| `egomed.exe`（onefile，78 MB） | **不需要**（GUI / 自检 / 诊断包 / 配置） | **交付方双击即用** | ✅ 2026-11-04 新增 |
| `python launcher.py <子命令>` | 需要 | 调试 | 保留 |

`egomed.exe` 与 `egomed.bat` **并存不冲突**：exe 只是入口层更薄，底层调的还是同一套 `launcher.py` 子命令。

## 二、exe 能做什么 / 不能做什么（**边界要说清，别误解**）

**能（零依赖，exe 内自带）**
- 双击打开**现场控制台**界面（tkinter）：菜单「文件 / 运行 / 工具 / 帮助」+ ⑥ 运行日志面板
- `doctor` / `check` 环境自检（缺什么、怎么补，一行一行说清）
- `diag` 生成诊断包（`versions.json` + 配置快照）
- 配置读写（`egomed_config.py`）、错误码人话化（`egomed_errors.py`）
- 读 `VERSION` 显示版本（**已实测**：`egomed 0.2.4 (contract v1.0.7)`）

**不能（推理依赖仍需服务端环境）**
- **不能**在 exe 里跑分割推理：torch 2.8 + CUDA + ultralytics + SAM2 + faster-whisper 合计 **> 2.5 GB**，
  塞不进单文件、也不该塞（且 CUDA DLL 依赖复杂，打出来也不可靠）。
- 所以**真跑分割仍需要目标机器有 egomed 环境**（或用便携包 + `安装环境.bat`）。
- exe 内置的 doctor 会明确区分"torch 不在 exe 内（正常）"与"环境真缺东西"。

> 这也是为什么 exe 只打包 78 MB 而不是 2.5 GB：**入口层零依赖，推理层走环境**。

## 三、怎么打（可重复执行）

```powershell
python scripts/build_exe.py                 # 默认：onefile + GUI 窗口，产出 dist/EgoMed.exe
python scripts/build_exe.py --onedir        # 打成目录（启动快，但要分发整个目录）
python scripts/build_exe.py --console       # 带控制台窗口（给 doctor 看输出）
python scripts/build_exe.py --zip           # 额外打一个分发用 zip
python scripts/build_exe.py --clean         # 先删旧产物再打
```

首次约 1~2 分钟（含 cv2/PIL）；已缓存时 ~30~60 秒。

## 四、打包时踩到的两个坑（**换机器/换环境会再遇到**）

### 坑 1：conda 环境的 Tcl/Tk 不在标准位置 → GUI 静默起不来

- 现象：exe 里 `import tkinter` 失败，doctor 报「E3002 tkinter 不可用（GUI 起不来）」；
  而**源码环境完全正常**（`TkVersion 8.6`、`Tk() OK`）。
- 根因：PyInstaller 的 hook 按**标准布局**（`<prefix>/tcl/tcl8.6`）找 Tcl/Tk，
  而 conda 把它放在 `Library/lib/tcl8.6`、`Library/lib/tk8.6` → **静默漏掉**。
- 修法（已写进 `scripts/build_exe.py`）：手动 `--add-data` 把两处目录塞进 `_tcl_data/`，并 `--collect-all tkinter`。

### 坑 2：conda 的 DLL 在 `Library\bin` → 依赖分析扫不到 → `DLL load failed`

- 现象：`ImportError: DLL load failed while importing _tkinter: 找不到指定的模块`；
  打包日志其实已经报了 `WARNING: Library not found: could not resolve 'tcl86t.dll'`（**警告容易被刷屏淹掉**）。
- 根因：PyInstaller 分析 `_tkinter.pyd` 依赖时只扫 conda 的 `DLLs` 目录，
  而 `tcl86t.dll` / `tk86t.dll` 在 `Library\bin` → 没被打进去。
- 修法：**调 PyInstaller 之前把 `Library\bin` 临时加进 `PATH`**（它的依赖分析用 PATH 找 DLL），
  再 `--add-binary` 显式补 tcl/tk 两个 DLL 双保险。

> 这两条都是**环境布局问题，不是代码问题** —— 换一台机器打包如果又出现 tkinter 起不来，先查这两条。

## 四之二、运行时坑：exe 双击「一闪就退」（2026-11-05 定位并修复）

**现象**：双击 `egomed.exe` 什么都不发生。而 `--version` 正常（退出码 0）、`doctor` 也全通过 ——
**唯独界面起不来**。GUI 版 exe 用 `runw.exe` bootloader **没有控制台，报错直接消失**，只能看到进程退出码 2。

**定位**：换成带控制台的构建（`build_exe.py --console`）跑一次，真因立刻现形：

```
egomed: error: argument cmd: invalid choice: 'D:\EgoMed-Agent\client\live_console.py'
```

根因是 `launcher.py` 里转发子脚本的 6 处写成 `subprocess.call([sys.executable, <脚本>, ...])`，
而**exe 模式下 `sys.executable` 指向 exe 自己** → 实际执行的是 `egomed.exe client/live_console.py`，
脚本路径被当成子命令喂回 argparse，报 invalid choice 后退出。

受影响：`console` / `demo` / `sim` / `diag` / `calibrate` / `regress` / `soak` / `check`（凡转发 .py 的全中招）；
`doctor` 不起子进程，所以完全没暴露。

**修法**：新增 `py_for_script()` 与 `run_script()` ——
源码模式仍用 `sys.executable`（行为与改前逐行一致），exe 模式改走 `find_python()`；
真找不到解释器时明确报 **E3001**（"开界面/自检/诊断包不需要 Python，跑脚本需要 egomed 环境"），不再静默崩。
顺带给子进程加 `CREATE_NO_WINDOW`：`runw` 版父进程无控制台时，子进程 python 会弹一个黑框，双击体验很差。

**实测（2026-11-05 19:25，重新打包后）**

| 项 | 结果 |
|---|---|
| `egomed.exe --version` | 退出码 0 |
| `egomed.exe doctor` | 退出码 0（自检全部通过） |
| 双击 `egomed.exe` | **12 秒**内弹出「EgoMed-Agent 现场控制台（N26） v0.2.4」窗口，中途无黑框 |
| 源码模式 `python launcher.py doctor` | 全部通过（行为与改前一致） |

> **教训**：`--version` / `doctor` 通过 ≠ exe 能用。这两条恰好都不走"转发子脚本"的路径，
> 会给出**假阳性**（11-04 那句"GUI 进程双击启动存活正常"就是这么误判的）。
> 验收 exe 至少要有两条：**双击能起 GUI**、**至少一个转发型子命令能跑**（`console` / `diag`）。

## 五、exe 模式下 `launcher.py` 改了什么（3 处适配）

| 位置 | 源码模式 | exe 模式（`sys.frozen`） |
|---|---|---|
| 项目根定位 | `Path(__file__).parent` | **`Path(sys.executable).parent`**（exe 所在目录）——因为 `_MEIPASS` 里没有 `client/` 也没有 `VERSION` |
| 版本号来源 | 根目录 `VERSION` | exe 旁边的 `VERSION`（+ `_MEIPASS` 兜底） |
| python 解释器查找 | conda / `EGOMED_PY` / PATH | **额外优先找 exe 同目录的 `python/`、`env/`、`runtime/`**；且**跳过 exe 自己**（否则会把 exe 再起一遍） |

实测（`dist/EgoMedCli3f.exe` 放在项目根）：

```
[--] EgoMed-Agent 环境自检  v0.2.4（契约 v1.0.7）
[--] 项目根：D:\EgoMed-Agent          ← 正确
[OK] 解释器：D:\ScienceApp\anaconda\envs\egomed\python.exe   ← 正确（没把自己当解释器）
[OK] GUI：tkinter 可用                ← 修好之后
```

## 六、给交付方的说法（一句话版）

> **双击 `egomed.exe` 就能打开控制台界面、跑自检、生成诊断包，不需要装 Python；
> 要真跑分割，需要本机有 egomed 环境（或用便携包 + `安装环境.bat` 一键装）。**
