"""uint16 bin shard 的读写与随机 batch 采样（nanoGPT 式原始二进制，无文件头）。

- write_shard：token 数组以 uint16 小端（<u2）原始字节落盘，自动建父目录；
- Shards：扫描目录下 {split}_*.bin，np.memmap 只读惰性映射（不占内存），
  get_batch 随机采样 (x, y)——x = tokens[i:i+T]，y = tokens[i+1:i+T+1]。

shard 文件由 scripts/prepare_data.py 的 ShardWriter 产出（train_000.bin … / val_000.bin）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

UINT16 = np.dtype("<u2")


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    """把 token 数组以 uint16 小端原始字节写入 path。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(tokens, dtype=UINT16)
    arr.tofile(path)


class Shards:
    """一个 split（train/val）的 shard 集合：memmap 只读映射 + 随机 batch 采样。"""

    def __init__(self, data_dir: str | Path, split: str):
        self.dir = Path(data_dir)
        self.split = split
        self.paths = sorted(self.dir.glob(f"{split}_*.bin"))
        if not self.paths:
            raise FileNotFoundError(
                f"{self.dir} 下未找到 {split}_*.bin，请先运行 scripts/prepare_data.py"
            )
        self._maps = [np.memmap(p, dtype=UINT16, mode="r") for p in self.paths]
        self._lens = np.array([m.size for m in self._maps], dtype=np.int64)
        self.total = int(self._lens.sum())

    def get_batch(
        self, batch: int, seq_len: int, device: str, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """随机采 batch 条长度为 seq_len 的序列，返回 (x, y)，y 为 x 右移一位。

        shard 按可用长度加权抽样（每个 token 起点等概率）；跨文档边界不特殊处理
        （语料已用 <|endoftext|> 拼接，EOT 本身即边界信号，惯例做法）。
        """
        need = seq_len + 1
        usable = np.maximum(self._lens - need, 0)
        if usable.sum() <= 0:
            raise ValueError(f"{self.split} shards 均短于 seq_len+1={need}")
        weights = usable / usable.sum()
        sis = rng.choice(len(self._maps), size=batch, p=weights)
        # x/y 分别写入独立连续数组：同一缓冲的 t[:, :-1]/t[:, 1:] 切片非连续，
        # 下游 view(-1) 会报错（勿合写同一 buf 再切片）
        bx = np.empty((batch, seq_len), dtype=np.int64)
        by = np.empty((batch, seq_len), dtype=np.int64)
        for b, si in enumerate(sis):
            off = int(rng.integers(0, self._lens[si] - need + 1))
            row = self._maps[si][off : off + need]
            bx[b] = row[:-1]
            by[b] = row[1:]
        x = torch.from_numpy(bx).to(device)
        y = torch.from_numpy(by).to(device)
        return x, y
