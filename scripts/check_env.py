"""TAIS Obsidian 训练/推理环境自检脚本。

对应《从零构建TAIS-Obsidian_总体实施计划.md》§1.2 检查清单。
用法：python scripts/check_env.py
全部断言通过即退出码 0。
"""
import sys

import torch

# Windows 控制台默认 GBK，避免中文输出乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


print(f"torch {torch.__version__} / wheel CUDA {torch.version.cuda}")
check("CUDA 可用", torch.cuda.is_available())

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"设备: {name} / compute capability {cap} / 显存 {vram_gb:.1f} GB")

    want_sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    # 关键检查：wheel 里是否有能在本机跑的 kernel。
    # CUDA 二进制兼容性：同一主版本内，低 minor 的 SASS 可在高 minor 设备上运行
    # （如 sm_86 的 cubin 可在 sm_89 Ada 上运行）；compute_XX PTX 可由驱动 JIT。
    def _compatible(arch: str) -> bool:
        if arch.startswith("sm_"):
            try:
                major, minor = int(arch[3]), int(arch[4:])
            except (IndexError, ValueError):
                return False
            return major == cap[0] and minor <= cap[1]
        if arch.startswith("compute_"):
            return True  # PTX 可 JIT
        return False

    compat = [a for a in arch_list if _compatible(a)]
    check("arch_list 含兼容 kernel", len(compat) > 0,
          f"want {want_sm} 或同主版本低 minor, got {arch_list}, 可用 {compat}")

    check("bf16 支持", torch.cuda.is_bf16_supported())

    # 实际跑一次 bf16 matmul，验证 tensor core 路径
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        s = (a @ a).sum().item()
        check("bf16 matmul 实测", s != 0 and s == s)  # 非 0 且非 NaN
    except Exception as e:  # noqa: BLE001
        check("bf16 matmul 实测", False, repr(e))

    free_gb, total_gb = [x / 1024**3 for x in torch.cuda.mem_get_info()]
    print(f"空闲显存: {free_gb:.1f} / {total_gb:.1f} GB")

if failures:
    print(f"\n{len(failures)} 项检查失败: {failures}")
    sys.exit(1)
print("\n全部检查通过。")
