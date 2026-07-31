"""把 latest.pt 训练断点导出为 save_pretrained 目录（config.json + model.safetensors bf16 + tokenizer.json）。

链路用途：Colab 训练产物 latest.pt → 本脚本导出 → 上传 HF → 下载后 from_pretrained 直接推理。
模型 config 从 ckpt["model_cfg"] 恢复（save_checkpoint 总是写入）；极旧格式无该字段时
回退 --config 训练 JSON（经 build_model_config 还原，语义与训练时一致）。

用法：
  python scripts/export_final.py --ckpt checkpoints/<run>/latest.pt --out checkpoints/<run>/final
  python scripts/export_final.py --ckpt <latest.pt> --out <final> --config configs/<cfg>.json  # 无 model_cfg 时
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# ipykernel OutStream 无 reconfigure 方法（notebook import 即炸），hasattr 守卫
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.train import build_model_config, copy_tokenizer_to_final  # noqa: E402


def model_cfg_from_ckpt_dict(d: dict) -> ModelConfig:
    """从 ckpt["model_cfg"]（model.config.__dict__ 快照）还原 ModelConfig（过滤未知键向后兼容）。"""
    valid = {f.name for f in dataclasses.fields(ModelConfig)}
    dropped = sorted(set(d) - valid)
    if dropped:
        print(f"[export] 忽略 model_cfg 未知字段（向后兼容）: {dropped}")
    return ModelConfig(**{k: v for k, v in d.items() if k in valid})


def export_final(ckpt_path: str | Path, out_dir: str | Path, config: str | Path | None = None) -> Path:
    """latest.pt → save_pretrained 目录（bf16 权重 + config.json + tokenizer.json），返回产物目录。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_cfg"):
        model_cfg = model_cfg_from_ckpt_dict(ckpt["model_cfg"])
        print(f"[export] 模型 config 从 ckpt['model_cfg'] 恢复（step={ckpt.get('step', '?')}）")
    elif config is not None:
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        model_cfg = build_model_config(cfg)
        print(f"[export] ckpt 无 model_cfg，回退 --config {config} 还原模型 config")
    else:
        raise SystemExit("[export] ckpt 无 model_cfg 字段且未给 --config，无法还原模型结构，拒绝导出")
    model = TaisObsidianForCausalLM(model_cfg)
    model.load_state_dict(ckpt["model"])  # strict：断点与结构不一致即报错（防静默错配导出）
    out = Path(out_dir)
    model.save_pretrained(out)
    copy_tokenizer_to_final(out)
    print(f"[done] 导出完成：{ckpt_path} → {out}（config.json + model.safetensors bf16 + tokenizer.json）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="latest.pt → save_pretrained 目录（随附 tokenizer.json）")
    ap.add_argument("--ckpt", required=True, help="训练断点 latest.pt 路径")
    ap.add_argument("--out", required=True, help="输出 save_pretrained 目录（如 checkpoints/<run>/final）")
    ap.add_argument("--config", default=None, help="训练 config JSON（仅 ckpt 无 model_cfg 时回退用）")
    args = ap.parse_args()
    export_final(args.ckpt, args.out, args.config)


if __name__ == "__main__":
    main()
