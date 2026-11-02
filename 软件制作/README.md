# 软件制作（计划与草稿区）

> 用途：软件化交付的**计划 + 草稿区**。`launcher.py` / `egomed.bat` / `config.toml` 已于 2026-10-29 迁到**仓库根目录**（增量 1 收尾），本目录只留计划、分工与派工单。
> 创建：2026-10-28 ｜ 维护：闫
> 上级设计：`docs/软件化交付流程.md`（v0.1 草案）｜任务表：`docs/总计划.md`

## 当前文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `计划_软件化实施.md` | 四个增量的目标、验收标准、排期、风险 | ✅ v0.1 |
| `分工_软件化.md` | 闫 / 李 的分工与文件属主 | ✅ |
| `PR派工单_李_软件化.md` | 给李的 PR 派工单（P1~P5） | ✅ |
| `README.md` | 本说明 | ✅ |

## 已迁出（现在在根目录）

| 文件 | 现在的位置 | 说明 |
|---|---|---|
| `launcher.py` | `./launcher.py` | 统一启动器 v0.2（9 个子命令，全部已实装） |
| `egomed.bat` | `./egomed.bat` | 唯一入口壳（解释器兜底 + 透传退出码） |
| `config.toml` | `./config.toml` | 唯一配置源（含 `[ui]` 段：上次界面选择） |
| `egomed_config.py` | `./egomed_config.py` | `config.toml` 的**唯一读写实现**（四层优先级 / 坏配置备份 / 带注释回写） |
| `VERSION` | `./VERSION` | 软件版本号**唯一来源**（`--version` 与 GUI「关于」同源） |

## 旧 bat 的现状（增量 1 收尾完成）

`启动模拟眼镜.bat` / `现场控制台.bat` / `环境自检.bat` 已改为**转发壳**，只 `call egomed.bat <子命令>`，
不再各自重复 6 行解释器候选路径；语义与原脚本一一对应，行为不变：

| 旧入口 | 现等价命令 | 下游脚本 |
|---|---|---|
| `启动模拟眼镜.bat` | `egomed.bat sim` | `client/launch_sim_glasses.py` |
| `现场控制台.bat` | `egomed.bat console` | `client/live_console.py` |
| `环境自检.bat` | `egomed.bat check` | `scripts/check_env.py` |

## 增量 2（GUI 菜单 + 运行日志）已完成 — 2026-11-02

`client/live_console.py` 加了菜单栏与 ⑥ 运行日志面板，长任务都能在界面里看实时输出：

| 菜单项 | 等价命令 | 下游脚本 |
|---|---|---|
| 工具 → 环境自检 | `egomed.bat doctor` | `launcher.py` 内置 |
| 工具 → 详细环境检查 | `egomed.bat check` | `scripts/check_env.py` |
| 运行 → 一键回归（mock 套件） | `egomed.bat regress --suite mock` | `client/run_regression.py` |
| 运行 → 一键回归（完整） | `egomed.bat regress` | `client/run_regression.py` |
| 运行 → 长稳测试（60 分钟） | `egomed.bat soak --minutes 60` | `client/run_soak.py` |
| 运行 → 停止长任务 | —（GUI 内） | 先 CTRL_BREAK，5s 不退再 taskkill /T |

## 增量 3（配置持久化）已完成 — 2026-11-02

`config.toml` 由 `egomed_config.py` 统一读写，优先级：**命令行 > 环境变量 > 文件 > 内置默认**。

| 能力 | 入口 |
|---|---|
| 记住上次界面选择 | 关窗口自动写 `[ui]`；下次打开自动恢复 |
| 改软件级配置 | GUI「文件 → 设置…」（服务端/超时/JPEG 质量/回显方式/日志级别） |
| 看/手改配置 | GUI「文件 → 打开配置文件」 |
| 丢弃改动重读 | GUI「文件 → 重新载入配置」 |
| 环境变量覆盖 | `EGOMED_SERVER_URL` / `EGOMED_FPS` / `EGOMED_RENDERER` / `EGOMED_LOG_LEVEL` |
| 配置写坏 | 自动改名 `config.bak-<时间戳>.toml` + 用默认值启动 + 一行提示，不拦启动 |
| 命令行查看生效值 | `python egomed_config.py` ／ 自测 `python egomed_config.py --selftest` |

## 验证方式（不需要服务端）

```powershell
# 配置模块自测：11 项（缺文件 / 往返 / 坏配置备份 / 环境变量 / 类型保真）
D:\ScienceApp\anaconda\envs\egomed\python.exe egomed_config.py --selftest

# 精简自检：10 项 + 退出码（0 通过 / 2 有问题）
D:\ScienceApp\anaconda\envs\egomed\python.exe launcher.py doctor

# 详细自检（原「环境自检.bat」的 14 项）
.\环境自检.bat

# 查看子命令
.\egomed.bat --help

# 演示流程预演（不起任何进程）
.\egomed.bat demo --dry-run
```

## 迁移判定（已满足）

- [x] 增量 1（launcher + 入口）验收通过：`doctor / console / calibrate / check / sim` 走通；
- [x] 根目录 bat 改为转发壳，且旧行为不变（`现场控制台.bat --selftest`、`环境自检.bat` 实测 EXIT=0）；
- [x] 写进 `docs/工作日志.md` 并提交。
