# 软件制作（工作区）

> 用途：软件化交付的**草稿区**。计划和骨架先放这里，验收通过后再迁到项目根目录。
> 创建：2026-10-28 ｜ 维护：闫
> 上级设计：`docs/软件化交付流程.md`（v0.1 草案）｜任务表：`docs/总计划.md`

## 为什么单独建这个目录

1. **不动现有可运行的东西**：`launcher.py` 稳定之前，根的 11 个 bat 与 `client/` 全部保持原样，随时可回退；
2. **草稿与成品分离**：这里的东西验收通过才移到根目录，避免半成品混进主链路；
3. **中文目录不影响运行**：本目录只有代码与文档，没有 `cv2.imread` 的图片数据（OpenCV 5.0 实测不支持中文路径，图片数据一律留在 `data/`、`runs/`）。

## 文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `计划_软件化实施.md` | 四个增量的目标、验收标准、排期、风险 | ✅ v0.1 |
| `launcher.py` | 统一启动器骨架（子命令 + 返回码 + 环境自检已实装） | ◐ 骨架可跑 |
| `egomed.bat` | 唯一入口壳（解释器兜底 + 调 launcher） | ◐ 骨架可跑 |
| `config.toml` | 配置模板（未被 launcher 强制依赖） | ✅ v0.1 |

## 验证方式（不需要服务端）

```powershell
# 环境自检：输出各检查项 + 退出码
D:\ScienceApp\anaconda\envs\egomed\python.exe launcher.py doctor

# 查看子命令
D:\ScienceApp\anaconda\envs\egomed\python.exe launcher.py --help

# 双击路径（bat 会自己找解释器）
egomed.bat doctor
```

## 迁移到根目录的条件

- 增量 1（launcher + 入口）验收通过：`doctor / diag / console / calibrate` 四条走通；
- 根目录 bat 改为转发壳，且旧行为不变；
- 写进 `docs/工作日志.md` 并提交。
