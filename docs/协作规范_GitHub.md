# 协作规范（GitHub）｜ v1，2026-10-11 起

> 依据：`docs/决策记录.md` **D15**（GitHub 唯一真源 + 短分支 + PR 门禁）｜ **D8**（单文件单负责人 + 指令式派活）
> 适用：闫（`client/**` + `docs/**`）· 李（`server/**`）
> 远端：`https://github.com/KissRain25/egomed-agent-demo.git`

---

## 1. 三条原则（先记住这三句）

1. **`main` 是唯一真源** —— 任何人手上都不该有"另一份最新版"。
2. **`main` 永远可交付** —— 判据是 **CI「mock 8/8」为绿**（不需要 GPU，几十秒出结果）。
3. **交付物必须带版本锚点** —— 每个 PR 填"**基于 commit**"，这一行能防住 90% 的分叉事故。

---

## 2. 分支模型（2 人最省事的一版）

| 分支 | 谁用 | 命名 | 生命周期 |
|------|------|------|---------|
| `main` | 所有人 | — | 常驻，受门禁保护 |
| 任务分支 | 李 | `server/t1-timeout`、`server/t5-log-rotate` | **≤2 天**，合完即删 |
| 任务分支 | 闫 | `client/n49-console`、`docs/weekly-plan` | 同上 |

**为什么不用长期分支**（如 `dev`）：2 人 + 文件边界清晰，长期分支只会制造"第二次分叉"。

---

## 3. 日常操作卡

### 3.1 开工前（**必须**）

```powershell
git pull --rebase
git status                    # 应当干净；有意外改动先弄明白
git rev-parse --short HEAD    # ★ 记下这个 hash —— PR 里要填
```

### 3.2 干活

```powershell
git switch -c server/t1-timeout        # 或 client/n49-console
# 改代码（只动自己负责的目录）
python -m py_compile server/api.py     # 至少语法过
回归.bat --suite mock                  # 快套件（约 1 分钟，不用 GPU）
```

### 3.3 收工 / 提交 PR

```powershell
git add <改动文件>                     # 不要 git add -A（避免误带垃圾）
git commit -m "feat(server): T1 并发超时清零 —— A2/A1 分离排队"
git push -u origin server/t1-timeout
# 然后在 GitHub 上开 PR（模板会自动带出，逐项填写）
```

### 3.4 合并

| 情形 | 谁能合 | 前置 |
|------|--------|------|
| 只动自己目录（`server/**` 只李、`client/**` 只闫） | **本人**即可合 | CI 绿 |
| **跨目录**（动了对方的文件） | **对方 review 后合** | CI 绿 + 例外登记（见 §6） |
| 改接口（契约） | **双方同意**后合 | 走 v1.x 流程 + 更新契约 |

### 3.5 网络/代理（你们踩过）

```powershell
# 需要代理时（只在本次命令生效，最安全）
git -c http.proxy=http://127.0.0.1:1080 -c https.proxy=http://127.0.0.1:1080 push origin main

# 或设为本地配置（只影响本仓库）
git config --local http.proxy http://127.0.0.1:1080
git config --local --unset http.proxy          # 不用了就撤掉

# 代理长期不稳 → 走 §7 的"内网 bare / 增量 bundle"
```

---

## 4. 合并门禁（CI）怎么看

| 现象 | 含义 | 怎么办 |
|------|------|--------|
| ✅ `mock 8/8（无需 GPU）` | 协议与边界都过 | 可以合 |
| ❌ 红 | **有东西坏了** | 点进日志看是哪一步（语法 / 服务端起不来 / C1 正常帧 / C2 坏帧） |
| ⚠️ `验收样例缺失` 警告 + 跳过 | 云端没有 `data/samples`，**门禁没真正生效** | 见 §5（要让样例入库），否则等于没门禁 |

CI 文件：`.github/workflows/regress-mock.yml`；依赖：`requirements-ci.txt`（**只装 mock 套件够用的包，不装 torch**）。

---

## 5. 素材与仓库体积（**重要，决定 CI 是否真的有效**）

现状：`.gitignore` 里 `/data/*`、`/runs/`、`/weights/`、`*.pt` 全被忽略 —— 这是对的（本机 `data` **72.5 GB**、`runs` **10.5 GB**）。

但这带来一个副作用：**验收样例（`data/samples/acdc`、`data/samples/bad`）不在仓库里**，而 `acceptance_test --suite mock` 需要它们 → **云端 CI 会缺素材**。

**建议（待定）**：只把**最小可复现样例**入库（约 8 张图、<1.5 MB）：

```
data/samples/acdc/ci_0001.jpg ~ ci_0003.jpg     # 3 张正常帧（C1 用）
data/samples/bad/*.jpg                          # 5 张坏帧（C2 用，可由 make_test_frames.py 生成）
```

`.gitignore` 补两条窄例外：

```
!/data/samples/acdc/ci_*.jpg
!/data/samples/bad/
```

好处：① CI 门禁真正生效；② 换机器/新人上手不必再"自己找样例"；③ 与 N29 交付打包口径一致。
代价：仓库 +1.5 MB（相对 13.6 MB 的 `.git` 可忽略）。

---

## 6. 例外流程（动了对方的文件时）

`server/**` 归李、`client/**` + `docs/**` 归闫。**要动对方的目录，必须走这条**（先例：D10 的服务端小修包）：

```
① 备份          server/api.py.bak-<月日>
② 改          只改必需部分
③ 跑回归       回归.bat（mock + real）
④ 登记         docs/工作日志.md + docs/决策记录.md（如需）
⑤ PR + review 开 PR，对方 review 后合
⑥ ★ 同步       把合并后的最新文件/commit 明确发给对方（补齐 D10 缺的这一步）
```

**第 ⑥ 步是 D15 新增的** —— 只"通知"不够，必须"**让对方拿到最新版**"。

---

## 7. 首次接入与降级方案

### 7.1 李首次接入（他手上不是仓库副本）

任选其一：

```powershell
# 方案 1：正常克隆（网络好时）
git clone https://github.com/KissRain25/egomed-agent-demo.git

# 方案 2：离线初始包（网络不通时，由闫生成）
#   闫：git bundle create runs\_share\egomed_<日期>.bundle --all      （约 12 MB，微信可传）
git clone egomed_<日期>.bundle egomed-agent
cd egomed-agent
git remote set-url origin https://github.com/KissRain25/egomed-agent-demo.git
```

接上后立刻验证：`python scripts/check_env.py` → `回归.bat --suite mock` → **8/8**。

### 7.2 降级（GitHub 长期不可用）

```powershell
# ① 内网 bare 仓库（局域网直连，最快）
#   闫：git init --bare D:\share\egomed.git; git remote add share D:\share\egomed.git; git push share main
#   李：git remote add share \\<主机名>\share\egomed.git; git pull share main

# ② 增量 bundle（只含新提交，几 KB～几 MB）
#   闫：git bundle create runs\_share\inc_<日期>.bundle <上次的hash>..main
#   李：git pull inc_<日期>.bundle main
```

> **绝对不要**用网盘/Syncthing 实时同步 `.git` 目录 —— 会损坏仓库。

---

## 8. 每次 PR 必须带的东西（清单）

- [ ] **基于 commit**（`git rev-parse --short HEAD`）
- [ ] 改动文件列表 + ≤5 行摘要
- [ ] **是否触碰接口**（是/否；是则附契约变更 + 已通知对方）
- [ ] 验收命令 + **数字**（改前 → 改后）
- [ ] 关键日志片段
- [ ] 遗留 / 风险 / 需要对方提供什么
- [ ] 只动了自己负责的目录（否则附例外登记）

---

## 9. 一页速记

```
main = 唯一真源，永远绿（CI：mock 8/8，不需要 GPU）
开工：git pull --rebase   →  记下 HEAD hash
干活：git switch -c server/xxx（≤2 天）
收工：commit → push → 开 PR（模板必填"基于 commit"）
合并：自己目录自己合；跨目录要对方 review；改接口要双方同意
例外：备份 → 改 → 回归 → 登记 → PR → ★ 把最新版同步给对方
不通：内网 bare / 增量 bundle（禁用网盘同步 .git）
```
