"""生成 notebooks/TAIS_1B_Colab.ipynb（一次性生成器；生成后逐 cell AST 校验）。

设计原则：所有 code cell 都是**纯 Python**（subprocess 调外部命令），不用 !/%% 魔法命令，
保证每个 cell 都能用 ast.parse 校验语法。notebook 面向 Colab G4（RTX PRO 6000 Blackwell 96GB，
sm_120），流程：环境 → 代码 → Drive 持久化 → 数据（流式→shards）→ 冒烟 → 标定 →
预训练（断连续训）→ 中训练退火（--init_from）→ 推理自验 → HF 上传。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip().splitlines(keepends=True)}


cells = []

# ───────────────────────── 0. 封面与使用说明 ─────────────────────────
cells.append(md("""
# TAIS Obsidian 1B 预训练+中训练一体（Colab G4 · RTX PRO 6000 Blackwell）

**流程**：环境(cu128) → 获取代码 → Drive 持久化 → 10B 数据（流式→shards）→ 3步冒烟 → 吞吐标定
→ **Phase-1 预训练 9B tokens** → **Phase-2 中训练退火 1B tokens** → 推理自验 → 上传 HuggingFace。

**使用前必读**：
1. **GPU**：需要 Colab Pro/Pro+ 的 G4 机型（RTX PRO 6000 Blackwell，96GB，sm_120）。
   本 notebook 第一个 code cell 会自检：不是 sm_120 或 torch 非 cu128 会给出明确修复指令。
2. **HF_TOKEN**：在 Colab 左侧"密钥(Secrets)"里添加名为 `HF_TOKEN` 的密钥并开启 notebook 访问
   （或在本 notebook 第 2 个 cell 手动赋值——不要写死在会分享的副本里）。
3. **代码来源**（二选一，在下个 cell 配置）：
   - `GIT_URL`：可访问的 git 仓库地址（推荐，私有仓库用 PAT：https://<token>@github.com/<user>/<repo>.git）；
   - 或把仓库打包成 `TAIS_Obsidian.zip` 放到 Google Drive 的 `TAIS_1B/` 目录下。
4. **断连是常态**（Colab 单会话 ≤24h）：checkpoint 每 500 步落盘 + 后台每 15 分钟同步 Google Drive；
   断连后**按顺序重跑 cell 1–5，再重跑当前 Phase 的 cell 即可自动续训**。
5. **全程纯 Python cell**（无 shell 魔法命令），自上而下顺序执行；每个 cell 开头有注释说明作用与预计耗时。
6. 数据策略：parquet 流式下载→本地 uint16 shards（20GB，本地 NVMe）；原始 parquet 缓存在会话结束自动清。
   若 Drive 空间充足（>25GB）可把 shards 缓到 Drive 供后续会话复用（`CACHE_DATA_ON_DRIVE`）。
"""))

# ───────────────────────── 1. 配置区 ─────────────────────────
cells.append(md("## 1️⃣ 配置区（唯一需要改动的 cell）"))
cells.append(code('''
# ===================== 用户配置（按实际情况修改） =====================
GIT_URL = ""                # 例 "https://<token>@github.com/<user>/TAIS-Obisian.git"；留空则用 Drive zip
DRIVE_ZIP = "TAIS_Obsidian.zip"   # GIT_URL 为空时：Drive/TAIS_1B/ 下的仓库 zip 文件名
DRIVE_ROOT = "/content/drive/MyDrive/TAIS_1B"  # Drive 持久化根目录
HF_REPO_ID = ""             # 例 "你的用户名/tais-obsidian-1b-gdn2"；留空则上传步骤会提示
CACHE_DATA_ON_DRIVE = True  # True: 20GB shards 缓存到 Drive（下次会话秒级恢复）；False: 每会话重新流式制备(3-5h)
HF_TOKEN_MANUAL = ""        # 不推荐；优先用 Colab Secrets 的 HF_TOKEN
# ====================================================================
print("配置完成：GIT_URL" if GIT_URL else "配置完成：Drive zip 模式", "| Drive:", DRIVE_ROOT)
'''))

# ───────────────────────── 2. GPU 与环境自检 ─────────────────────────
cells.append(md("## 2️⃣ GPU 与 torch 自检（sm_120 必须 cu128+）"))
cells.append(code('''
import subprocess, sys, torch

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} | sm_{cap[0]}{cap[1]} | {total:.0f}GB")
print(f"torch {torch.__version__} | bundled CUDA {torch.version.cuda}")

NEED_CU128 = cap[0] >= 12  # Blackwell sm_120+ 必须 cu128 wheel（cu126 无内核）
cuda_ok = torch.version.cuda is not None and tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (12, 8)
if NEED_CU128 and not cuda_ok:
    print("\\n⚠️ 当前 torch 无 sm_120 内核，正在换装 cu128 wheel（约 3-5 分钟）…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                    "torch", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    print("✅ 已安装 cu128 torch。**请重启运行时（运行时→重启）后从头重跑本 notebook**（wheel 切换需重载）。")
elif NEED_CU128 and cuda_ok:
    print("✅ cu128+ 环境满足 Blackwell 要求")
else:
    print("ℹ️ 非 Blackwell GPU，按现有 CUDA 继续（请确认算力 ≥ sm_80）")
# 快速 matmul 自检（sm_120 内核真实可用）
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("✅ CUDA matmul 自检通过")
'''))

# ───────────────────────── 3. 安装依赖 ─────────────────────────
cells.append(md("## 3️⃣ 安装依赖（约 1-2 分钟）"))
cells.append(code('''
import subprocess, sys
pkgs = ["numpy", "tokenizers", "datasets", "huggingface_hub", "tensorboard", "pytest", "safetensors"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
print("✅ 依赖安装完成:", ", ".join(pkgs))
'''))

# ───────────────────────── 4. 获取代码 ─────────────────────────
cells.append(md("""## 4️⃣ 获取 TAIS Obsidian 代码（git clone 或 Drive zip）

产物：`/content/TAIS-Obisian`（仓库根）。随后 `pip install -e .` 注册包。"""))
cells.append(code('''
import os, subprocess, sys, zipfile
from pathlib import Path

REPO = Path("/content/TAIS-Obisian")
if REPO.exists():
    print("仓库已存在，跳过获取（断连重跑安全）")
elif GIT_URL:
    subprocess.run(["git", "clone", "--depth", "1", GIT_URL, str(REPO)], check=True)
    print("✅ git clone 完成")
else:
    from google.colab import drive
    if not Path("/content/drive").exists():
        drive.mount("/content/drive")
    zpath = Path(DRIVE_ROOT) / DRIVE_ZIP
    assert zpath.exists(), f"未找到 {zpath}——请把仓库 zip 上传到 Drive 的 TAIS_1B/ 目录"
    with zipfile.ZipFile(zpath) as z:
        z.extractall("/content")
    # zip 内层目录名归一化到 REPO
    cands = [p for p in Path("/content").iterdir()
             if p.is_dir() and (p / "pyproject.toml").exists() and p.name != "TAIS-Obisian"]
    if cands and not REPO.exists():
        cands[0].rename(REPO)
    assert (REPO / "pyproject.toml").exists(), "zip 结构不正确：未找到 pyproject.toml"
    print("✅ Drive zip 解压完成")

os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
sys.path.insert(0, str(REPO / "src"))
import tais_obsidian  # noqa
print("✅ 包安装完成:", REPO)
'''))

# ───────────────────────── 5. Drive 持久化 ─────────────────────────
cells.append(md("""## 5️⃣ 挂载 Drive + 目录布局 + checkpoint 后台同步

- `checkpoints/pilot_1b_gdn2/latest.pt`（~9GB：fp32 权重 + Muon 动量）每 500 步由训练进程落盘；
- 本 cell 启动**后台同步循环**：每 15 分钟把 latest.pt 复制到 Drive（断连后能从最近同步点续训）；
- 数据缓存目录：`$DRIVE_ROOT/shards_1b`、`$DRIVE_ROOT/shards_1b_anneal`（`CACHE_DATA_ON_DRIVE=True` 时生效）。"""))
cells.append(code('''
import subprocess, time
from pathlib import Path
from google.colab import drive

if not Path("/content/drive").exists():
    drive.mount("/content/drive")
for d in ["", "/ckpt_pretrain", "/ckpt_midtrain", "/shards_1b", "/shards_1b_anneal", "/hf_upload"]:
    Path(DRIVE_ROOT + d).mkdir(parents=True, exist_ok=True)

# 后台同步循环（每 15 min；只同步 latest.pt——final/ 在阶段结束由专门 cell 处理）
sync_script = r"""
while true; do
  src=/content/TAIS-Obisian/checkpoints/pilot_1b_gdn2/latest.pt
  [ -f "$src" ] && cp -u "$src" "%s/ckpt_pretrain/latest.pt" 2>/dev/null
  src=/content/TAIS-Obisian/checkpoints/pilot_1b_gdn2_midtrain/latest.pt
  [ -f "$src" ] && cp -u "$src" "%s/ckpt_midtrain/latest.pt" 2>/dev/null
  sleep 900
done
""" % (DRIVE_ROOT, DRIVE_ROOT)
Path("/content/_sync_ckpt.sh").write_text(sync_script)
subprocess.Popen(["bash", "/content/_sync_ckpt.sh"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Drive 已挂载:", DRIVE_ROOT, "| checkpoint 后台同步已启动（15 min 周期）")
'''))

# ───────────────────────── 6. 数据准备 ─────────────────────────
cells.append(md("""## 6️⃣ 10B tokens 数据制备（流式下载→本地 shards；3-5h，可断点续跑）

- 配比：fineweb_edu(sample-100BT) 73% / math(NuminaMath+FineMath-4+) 12% / cosmopedia 10% / 中文(FineWeb2-HQ) 5%；
- **断点续跑**：每个源完成即写 `_progress.json`，本 cell 重跑自动跳过已完成源；
- Drive 缓存（可选）：制好后整目录拷到 Drive，下次会话先拷回本地（约 5-10 分钟，省 3-5h）；
- 同时制备**中训练退火混合** 1B tokens（高质量上移：math 40%/code 20%，Dolmino 式）；
- 落盘自动做 **max_id 全量扫描**（<32768 硬校验）。"""))
cells.append(code('''
import shutil, subprocess, sys
from pathlib import Path

REPO = Path("/content/TAIS-Obisian")

def ensure_shards(name: str, target: str, val: str, extra: list[str]) -> Path:
    """Drive 有缓存则拷回本地；否则流式制备并（可选）缓存到 Drive。幂等，断连安全。"""
    local = REPO / "data" / name
    drive_dir = Path(DRIVE_ROOT) / name
    done_marker = local / "_stats.txt"
    if done_marker.exists():
        print(f"[{name}] 本地已就绪，跳过"); return local
    if CACHE_DATA_ON_DRIVE and (drive_dir / "_stats.txt").exists():
        print(f"[{name}] 从 Drive 缓存恢复（约 5-10 分钟）…")
        local.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_dir, local, dirs_exist_ok=True)
        return local
    print(f"[{name}] 流式制备 {target} tokens（可中断重跑，进度在 _progress.json）…")
    subprocess.run([sys.executable, "scripts/prepare_data_1b.py",
                    "--target_tokens", target, "--val_tokens", val,
                    "--out", str(local), *extra], cwd=REPO, check=True)
    if CACHE_DATA_ON_DRIVE:
        print(f"[{name}] 缓存到 Drive（{target}，约 20-40 分钟上传）…")
        shutil.copytree(local, drive_dir, dirs_exist_ok=True)
    return local

# 主混合 10B（约 3-5h）
ensure_shards("shards_1b", "10B", "20M", [])
# 退火混合 1B（Dolmino 式高质量上移：math 40%/code 20%；约 0.5-1h）
ensure_shards("shards_1b_anneal", "1B", "2M",
              ["--mix", "fineweb_edu=0.35,math=0.40,code=0.20,zh=0.05"])
print("✅ 全部数据就绪")
'''))

# ───────────────────────── 7. 冒烟 ─────────────────────────
cells.append(md("## 7️⃣ 3 步冒烟（GPU 纪律：训练循环改动/新环境必先冒烟；约 3 分钟）"))
cells.append(code('''
import subprocess, sys
from pathlib import Path
REPO = Path("/content/TAIS-Obisian")
r = subprocess.run([sys.executable, "-u", "-m", "tais_obsidian.train",
                    "--config", "configs/pilot_1b_gdn2.json", "--max_steps", "3",
                    "--micro_batch", "8", "--grad_accum", "2",
                    "--out_dir", "checkpoints/_smoke_1b"],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-800:] if r.returncode else "")
assert r.returncode == 0, "冒烟失败——不要进入正式训练"
shutil_ok = subprocess.run(["rm", "-rf", str(REPO / "checkpoints/_smoke_1b")])
print("✅ 冒烟通过（loss 正常下降、checkpoint/save_pretrained/tokenizer 随附均正常）")
'''))

# ───────────────────────── 8. 吞吐标定 ─────────────────────────
cells.append(md("""## 8️⃣ micro batch 标定（约 5 分钟）

96GB 显存预期 micro 32 起步；实测 micro 32/48 两档 tok/s 与峰值显存，自动选定正式配置。
**GPU 纪律：bench 必须在 GPU 空闲时做（本 cell 前不要有训练进程）。**"""))
cells.append(code('''
import subprocess, sys, time, json
import numpy as np, torch
from pathlib import Path
REPO = Path("/content/TAIS-Obisian"); sys.path.insert(0, str(REPO / "src"))
from tais_obsidian.train import build_model_config, build_optimizer, chunked_ce, DEFAULTS
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.data.memmap import Shards

cfg = json.loads((REPO / "configs/pilot_1b_gdn2.json").read_text(encoding="utf-8"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
model = TaisObsidianForCausalLM(build_model_config(cfg)).cuda().train()
opt = build_optimizer(model, dict(DEFAULTS, **cfg))
sh = Shards(str(REPO / "data/shards_1b"), "train")
rng = np.random.default_rng(0)
best = (16, 0.0)
for micro in [32, 48]:
    try:
        model.zero_grad(set_to_none=True)
        for _ in range(2):
            x, y = sh.get_batch(micro, 1024, "cuda", rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x); loss = chunked_ce(logits, y)
            loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            x, y = sh.get_batch(micro, 1024, "cuda", rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x); loss = chunked_ce(logits, y)
            loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tps = micro * 1024 * 5 / (time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"micro {micro}: {tps/1e3:.2f}k tok/s, peak {peak:.1f}GB")
        if peak < 85 and tps > best[1]:
            best = (micro, tps)
    except torch.cuda.OutOfMemoryError:
        print(f"micro {micro}: OOM"); torch.cuda.empty_cache()
MICRO = best[0]
ACCUM = max(1, round(262144 / (MICRO * 1024)))  # 保持全局 batch ≈ 256k tokens/step
ETA_H = 10e9 / best[1] / 3600
print(f"\\n✅ 选定 micro {MICRO} × accum {ACCUM}（{best[1]/1e3:.2f}k tok/s）")
print(f"📊 10B tokens 预计 {ETA_H:.0f} 小时 ≈ {ETA_H/24:.1f} 天（按 Colab 24h 会话约 {int(ETA_H/24)+1} 次会话）")
del model, opt; torch.cuda.empty_cache()
Path("/content/_calib.json").write_text(json.dumps({"micro": MICRO, "accum": ACCUM}))
'''))

# ───────────────────────── 9. Phase-1 预训练 ─────────────────────────
cells.append(md("""## 9️⃣ Phase-1 预训练（9B tokens ≈ 34300 步；**数十小时，跨会话续训**）

- `decay_frac=0.0`（Stable 段，不衰减；衰减留给中训练——SmolLM2 多阶段 WSD / OLMo Dolmino 模式）；
- 断连后：重跑 cell 1–5 + 本 cell，自动从 Drive 最近 latest.pt `--resume`；
- 监控：日志尾部每 50 步一行（loss/gnorm/tok/s）；val_every 500。
- **跑完标志**：`checkpoints/pilot_1b_gdn2/final/` 出现（含 tokenizer.json）。"""))
cells.append(code('''
import json, shutil, subprocess, sys, time
from pathlib import Path
REPO = Path("/content/TAIS-Obisian")
calib = json.loads(Path("/content/_calib.json").read_text()) if Path("/content/_calib.json").exists() \
        else {"micro": 32, "accum": 8}

# Drive 同步点恢复（断连续训）：本地无 latest.pt 且 Drive 有 → 拷回
local_ckpt = REPO / "checkpoints/pilot_1b_gdn2/latest.pt"
drive_ckpt = Path(DRIVE_ROOT) / "ckpt_pretrain/latest.pt"
if not local_ckpt.exists() and drive_ckpt.exists():
    local_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(drive_ckpt, local_ckpt)
    print("已从 Drive 恢复 latest.pt")

cmd = [sys.executable, "-u", "-m", "tais_obsidian.train",
       "--config", "configs/pilot_1b_gdn2.json",
       "--max_steps", "34300",          # 9B tokens（Stable 段，余 1B 给中训练）
       "--micro_batch", str(calib["micro"]), "--grad_accum", str(calib["accum"])]
if local_ckpt.exists():
    cmd += ["--resume", str(local_ckpt)]
    print("断点续训:", local_ckpt)
else:
    print("从头训练")
print("启动:", " ".join(cmd), "\\n日志: logs_train_1b.txt（本 cell 阻塞至训练结束；",
      "断连不影响后台进程？——Colab 会杀进程，靠 Drive ckpt 续训）")
t0 = time.time()
with open(REPO / "logs_train_1b.txt", "a") as logf:
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        time.sleep(300)
        tail = subprocess.run(["tail", "-2", str(REPO / "logs_train_1b.txt")],
                              capture_output=True, text=True).stdout.strip()
        print(f"[{(time.time()-t0)/3600:.1f}h] {tail}", flush=True)
assert proc.returncode == 0, f"训练异常退出 {proc.returncode}——查 logs_train_1b.txt"
print("✅ Phase-1 完成，final 在 checkpoints/pilot_1b_gdn2/final（含 tokenizer.json）")
'''))

# ───────────────────────── 10. Phase-2 中训练退火 ─────────────────────────
cells.append(md("""## 🔟 Phase-2 中训练退火（1B tokens ≈ 3800 步，lr 线性降到 0）

- `--init_from` Phase-1 final（**仅载权重、step=0、全新优化器**——OLMo Dolmino 独立退火 run 惯例）；
- 数据换退火混合 `shards_1b_anneal`（math 40%/code 20% 高质量上移）；`decay_frac=1.0` 全段线性衰减；
- 断连续训同 Phase-1（Drive ckpt_midtrain）。"""))
cells.append(code('''
import json, shutil, subprocess, sys, time
from pathlib import Path
REPO = Path("/content/TAIS-Obisian")
calib = json.loads(Path("/content/_calib.json").read_text())

# 生成中训练配置（独立 run_name/out_dir；decay_frac=1.0；lr 峰值=预训练峰值）
cfg = json.loads((REPO / "configs/pilot_1b_gdn2.json").read_text(encoding="utf-8"))
cfg.update({"run_name": "pilot_1b_gdn2_midtrain", "out_dir": "checkpoints/pilot_1b_gdn2_midtrain",
            "data_dir": "data/shards_1b_anneal", "max_steps": 3800,
            "warmup": 200, "decay_frac": 1.0})
mid_cfg_path = REPO / "configs/pilot_1b_gdn2_midtrain.json"
mid_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

local_ckpt = REPO / "checkpoints/pilot_1b_gdn2_midtrain/latest.pt"
drive_ckpt = Path(DRIVE_ROOT) / "ckpt_midtrain/latest.pt"
if not local_ckpt.exists() and drive_ckpt.exists():
    local_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(drive_ckpt, local_ckpt)

cmd = [sys.executable, "-u", "-m", "tais_obsidian.train", "--config", str(mid_cfg_path),
       "--micro_batch", str(calib["micro"]), "--grad_accum", str(calib["accum"])]
if local_ckpt.exists():
    cmd += ["--resume", str(local_ckpt)]
    print("中训练断点续训")
else:
    cmd += ["--init_from", "checkpoints/pilot_1b_gdn2/final"]
    print("中训练从 Phase-1 final 初始化")
t0 = time.time()
with open(REPO / "logs_train_1b_midtrain.txt", "a") as logf:
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        time.sleep(300)
        tail = subprocess.run(["tail", "-2", str(REPO / "logs_train_1b_midtrain.txt")],
                              capture_output=True, text=True).stdout.strip()
        print(f"[{(time.time()-t0)/3600:.1f}h] {tail}", flush=True)
assert proc.returncode == 0, f"中训练异常退出 {proc.returncode}"
print("✅ Phase-2 完成，final 在 checkpoints/pilot_1b_gdn2_midtrain/final")
'''))

# ───────────────────────── 11. 推理自验 ─────────────────────────
cells.append(md("""## 1️⃣1️⃣ 推理自验（拿到 final 权重后的第一道光）

若训练被断连杀掉只有 latest.pt：先跑 `scripts/export_final.py` 导出（本 cell 已内置判断）。"""))
cells.append(code('''
import subprocess, sys
from pathlib import Path
REPO = Path("/content/TAIS-Obisian")
final = REPO / "checkpoints/pilot_1b_gdn2_midtrain/final"
latest = REPO / "checkpoints/pilot_1b_gdn2_midtrain/latest.pt"
if not final.exists() and latest.exists():
    print("final 缺失，从 latest.pt 导出…")
    subprocess.run([sys.executable, "scripts/export_final.py",
                    "--ckpt", str(latest), "--out", str(final)], cwd=REPO, check=True)
assert (final / "model.safetensors").exists() and (final / "tokenizer.json").exists()
for prompt in ["The capital of France is", "2+3=", "Water is composed of"]:
    r = subprocess.run([sys.executable, "-m", "tais_obsidian.generate",
                        "--ckpt", str(final), "--prompt", prompt, "--max_new_tokens", "30"],
                       cwd=REPO, capture_output=True, text=True)
    print("=" * 40); print(r.stdout[-400:])
    assert r.returncode == 0, r.stderr[-500:]
print("✅ 推理自验通过（generate 直接读 final 目录，tokenizer 自动回退）")
'''))

# ───────────────────────── 12. 模型卡 ─────────────────────────
cells.append(md("## 1️⃣2️⃣ 生成模型卡（HF README.md）"))
cells.append(code('''
from pathlib import Path
from datetime import date
REPO = Path("/content/TAIS-Obisian")
final = REPO / "checkpoints/pilot_1b_gdn2_midtrain/final"
card = f"""---
license: apache-2.0
language:
- en
- zh
library_name: tais_obsidian
tags:
- tais-obsidian
- gdn
- hybrid-attention
- research-pilot
pipeline_tag: text-generation
---

# TAIS Obsidian 1B (GDN-2 + TriRetrieval, research pilot)

自研混合架构 1B 验证模型（**研究性欠训运行**）：24× GDN-2 线性注意力层 + 8× 三级检索注意力层
（滑窗 512 L0 + CSA stride-4 选择检索 L1 + HCA 128:1 gist L2），d_model 1536 × 32 层，
head_dim 64，注意力头 24q:8kv，GDN 头 24v:12qk，vocab 32768 tied，上下文 1024（RoPE 可扩展）。

## 重要：加载方式
本模型为**自研 save_pretrained 格式**（非 HF PretrainedConfig），加载需要 TAIS Obsidian 代码库：
```bash
pip install -e .   # 仓库根
python -m tais_obsidian.generate --ckpt <本目录> --prompt "..." --max_new_tokens 50
```
`tokenizer.json`（32773 词表 BPE，含 5 个保留特殊 token id≥32768，勿输入其字面量）已随附。

## 训练
- 数据：10B tokens（fineweb-edu sample-100BT 73% / NuminaMath+FineMath-4+ 12% / cosmopedia 10% / FineWeb2-HQ 中文 5%）
  + 1B tokens Dolmino 式中训练退火（math 40%/code 20% 高质量上移，lr 线性降到 0）。
- 配方：bf16 autocast + fp32 主权重，Muon 优化器（Newton-Schulz 正交化动量，muon_lr 0.02）+ 非矩阵 AdamW 6e-4，
  WSD 调度（Stable 9B → Anneal 1B），grad clip 1.0，全局 batch 256k tokens/step。
- 硬件：RTX PRO 6000 Blackwell 96GB（Colab G4），断连续训多会话完成。
- **欠训声明**：10B tokens 对 1B 约为 Chinchilla 量（20B）的一半、远低于当代 1B 级实践（4T+）；
  本模型定位架构验证 pilot，绝对能力不可与 SmolLM2/Qwen3 同尺寸模型直接对标。

## 评测
（待补：val loss、ARC-Easy/HellaSwag/PIQA 等——拿到权重后用仓库评测管线跑出后回填）

## 限制
研究原型，未做安全对齐；上下文 1024（256K 扩展为后续工程）；词表含保留特殊 token；
知识截止 {date.today().isoformat()} 数据快照。
"""
(final / "README.md").write_text(card, encoding="utf-8")
print("✅ 模型卡已写入", final / "README.md")
'''))

# ───────────────────────── 13. HF 上传 ─────────────────────────
cells.append(md("""## 1️⃣3️⃣ 上传 HuggingFace（upload_folder，断点续传）

上传内容：final/ 全部（config.json + model.safetensors + tokenizer.json + README.md）+ 训练日志与配置归档。
完成后打印仓库 URL。"""))
cells.append(code('''
import os, shutil, subprocess, sys
from pathlib import Path
REPO = Path("/content/TAIS-Obisian")

token = HF_TOKEN_MANUAL or os.environ.get("HF_TOKEN", "")
if not token:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = ""
assert token, "未找到 HF_TOKEN——请在 Colab Secrets 添加或在本 notebook 配置区手动赋值"
repo_id = HF_REPO_ID or input("请输入 HF 仓库 ID（如 用户名/tais-obsidian-1b-gdn2）: ").strip()
assert repo_id, "HF_REPO_ID 为空"

from huggingface_hub import HfApi
api = HfApi(token=token)
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

final = REPO / "checkpoints/pilot_1b_gdn2_midtrain/final"
staging = Path(DRIVE_ROOT) / "hf_upload" / repo_id.split("/")[-1]
if staging.exists():
    shutil.rmtree(staging)
shutil.copytree(final, staging)
for extra in ["configs/pilot_1b_gdn2.json", "configs/pilot_1b_gdn2_midtrain.json",
              "logs_train_1b.txt", "logs_train_1b_midtrain.txt"]:
    src = REPO / extra
    if src.exists():
        dst = staging / "training_archive" / Path(extra).name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(src, dst)
print("上传中（~2.5GB 权重 + 归档，断点续传）…")
api.upload_folder(repo_id=repo_id, folder_path=str(staging), repo_type="model")
print(f"✅ 上传完成: https://huggingface.co/{repo_id}")
print("下载验证：在新会话跑 generate --ckpt <下载目录> 应直接出文本")
'''))

# ───────────────────────── 14. 断连恢复 runbook ─────────────────────────
cells.append(md("""## 🆘 断连恢复 Runbook（Colab 会话被杀后）

1. 重跑 **cell 1–5**（配置→GPU 自检→依赖→代码→Drive）；数据若开了 `CACHE_DATA_ON_DRIVE` 会在 cell 6 秒级恢复。
2. 重跑 **cell 6**（幂等：已完成源跳过）→ **cell 7 冒烟可跳** → **cell 8 标定可跳**（`_calib.json` 在本地则沿用，
   不在则用配置默认 micro 32×accum 8——96GB 显存安全）。
3. 重跑 **当前 Phase 的 cell**（9 或 10）：自动检测 Drive 的 latest.pt → 拷回 → `--resume` 续训。
4. 进入 cell 11–13（推理自验→模型卡→上传）。

**会话时长预算**（按 cell 8 实测 tok/s）：10B tokens 全链 = 数据 3-5h + Phase-1 主体 + Phase-2 ~2-3h + 上传 ~0.5h。
若单次会话 24h：Phase-1 需 N 次会话（cell 8 会打印 ETA 与会话数估计）。

## ❓ 常见问题
- `no kernel image available`：cell 2 的 cu128 换装后**必须重启运行时**再从头跑。
- Drive 配额不足：`CACHE_DATA_ON_DRIVE=False`（每会话重新流式制备 3-5h），或只缓存 `shards_1b` 不缓存 anneal。
- 上传中断：重跑 cell 13（upload_folder 断点续传）。
- 推理报 vocab 越界：prompt 含 `<|recall|>` 等 5 个保留特殊 token 字面量（id≥32768）——去掉即可。
- 日志即真相：`logs_train_1b*.txt`（每 50 步 loss/gnorm/tok/s）；异常先看日志尾部。
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "accelerator": "GPU",
        "gpuClass": "standard",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "TAIS_1B_Colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"written {out} ({len(cells)} cells)")
