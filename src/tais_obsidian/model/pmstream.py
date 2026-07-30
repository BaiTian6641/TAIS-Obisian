"""PM-stream：mHC（Manifold-Constrained Hyper-Connections）式多流残差（arXiv:2512.24880）。

残差流由单流扩展为 n 条并行流（本项目 n=5：4 条内容流 + 1 条感知-记忆流 PM-stream，
见设计文档 §12.2/§13.4；4+1 分配与读写纪律为本项目独创设想，尚无实验背书）。
每个子层（mixer / mlp 各算一个 mHC "层"，对齐原文 Fig.3 将 Attention/FFN 展开计层）
按 mHC 原文 Eq.3 更新流状态 S ∈ R^{n×d}：

    S ← H_res·S + H_postᵀ·F(H_pre·S)                （arXiv:2512.24880 Eq.3）

混合系数参数化（同文 Eq.7/8，全部在 fp32 内计算）：

    x⃗'  = RMSNorm(vec(S))                            （Eq.7：对 n·d 维 vec 归一）
    H̃_pre  = α_pre ·(x⃗'φ_pre) + b_pre
    H̃_post = α_post·(x⃗'φ_post) + b_post
    H̃_res  = α_res ·mat(x⃗'φ_res) + b_res
    H_pre = σ(H̃_pre)；H_post = 2σ(H̃_post)；H_res = SinkhornKnopp(H̃_res)   （Eq.8）

Sinkhorn-Knopp（Eq.9）：M⁽⁰⁾ = exp(H̃_res)，迭代 M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))（先列归一后行归一）
共 t_max=20 次（附录 A.1），把 H_res 投影到 Birkhoff 多胞形（双随机矩阵，Eq.6）。
双随机矩阵谱范数 ≤1 且对乘法封闭，故跨层复合映射保持信号守恒，抑制无约束 HC 的
信号放大（原文 27B 实测 Amax 增益峰值 ~3000× → ~1.6×）。σ/2σ 为 H_pre/H_post 的
非负约束（§4.1 末段；×2 使 H_post=1 落在值域内）。

恒等初始化（★此处为推断实现★：mHC 原文未给出 bias 初始化细节，附录 A.1 仅给出
门控因子 α init=0.01；HC §2.3 的静态初始化 A_m=e_{k mod n}、A_r=I 位于约束流形边界，
σ/Sinkhorn 的内点映射无法精确表示，故采用下述同样满足"恒等等价于 pre-norm 残差"
目标的流形内点初始化）：

    φ = 0            （HC §2.3：动态参数 0 初始化，arXiv:2409.19606 Eq.11–13）
    b_pre = logit(1/n)·1  →  H_pre = (1/n)·1，Σ_j H_pre[j] = 1
    b_post = 0       →  H_post = 2σ(0) = 1
    b_res = 0        →  H_res = Sinkhorn(0) = (1/n) 均匀矩阵（双随机，行和=1）
    流初始化 = 嵌入复制 n 份（HC §2.1：H⁰ = (h⁰,...,h⁰)ᵀ）

可证：所有流恒等 S[j]=x_l ⇒ 子层输入 = Σ(1/n)x_l = x_l、写回系数 = 1、
行和为 1 的流混合保持各流相等——前向严格恒等于单流 pre-norm 残差
（tests/test_pmstream.py 判据：logits 逐点 diff < 1e-6）。

与 HC/mHC 原文的两处偏离（均为满足恒等初始化判据，已在测试中验证）：

1. 输出聚合用**均值**而非 HC §2.1 的求和：RMSNorm 的 eps 破坏严格尺度不变性，
   对 n× 幅度的求和结果做 final norm 会引入 ~1e-3 相对误差、破坏恒等判据；
   均值在恒等初始化下与单流基线严格一致，训练动态上与求和仅差常数因子 1/n，
   被 final RMSNorm 吸收。
2. 未采用 HC §2.3 的输出层权重 1/√n 缩放：它会改变子层输出幅度，
   与"同基础权重下逐点一致"判据直接冲突。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinkhorn_knopp(h_res: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """Sinkhorn-Knopp 投影到 Birkhoff 多胞形（arXiv:2512.24880 Eq.9）。

    M⁽⁰⁾ = exp(H̃_res)，随后迭代 M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))（先列归一、后行归一），
    共 t_max 次（原文附录 A.1 取 t_max=20）。输入 [..., n, n]，输出同形状双随机矩阵。
    调用方须保证 fp32（本函数内部再 float() 兜底）；全程可微（autograd 直传）。

    迭代数（吞吐优化 P0，2026-07-30）：t_max 由调用方经 PMStreamMix.t_max 控制。
    实测（sm_120，h_res 训练态偏离）：t_max=10 时双随机偏差 ≤1.3e-2、谱范数=1.0
    （信号守恒红线不破），吞吐 ×1.7 vs t_max=20；t_max=20（默认）= 原文精确语义。
    注：曾尝试 tol 早停（双随机偏差 <tol 即停），但 `.item()` 判定每次迭代同步 GPU，
    实测比固定 20 次更慢（4.2ms vs 1.5ms）——早停收益被同步开销抵消，故弃用，
    改"调小固定迭代数"路径（无同步、纯 GPU 流水）。
    """
    m = h_res.float().exp()
    for _ in range(t_max):
        m = m / m.sum(dim=-2, keepdim=True)  # T_c：列归一
        m = m / m.sum(dim=-1, keepdim=True)  # T_r：行归一
    return m


class PMStreamMix(nn.Module):
    """单个子层的 mHC 混合系数生成器（H_pre/H_post/H_res，逐 token，fp32）。

    参数（fused 布局，对齐原文 Eq.10–13 将 φ/b 合并的记法）：
      phi  [n·d, n²+2n]：动态投影 (φ_pre | φ_post | φ_res)，0 初始化（HC §2.3）；
      bias [n²+2n]：静态偏置 (b_pre | b_post | b_res)，恒等初始化见模块 docstring；
      alpha_pre/post/res：标量门控 α，init 0.01（附录 A.1）。
    """

    def __init__(
        self,
        d_model: int,
        n_stream: int,
        eps: float = 1e-6,
        t_max: int = 20,
        constrain: bool = True,
    ):
        super().__init__()
        assert n_stream >= 2, f"PM-stream 流数须 ≥2，实际 {n_stream}"
        self.n = n_stream
        self.d = d_model
        self.eps = eps
        # Sinkhorn 迭代数（吞吐优化 P0）：默认 20=原文精确语义（向后兼容，恒等判据/
        # 旧 checkpoint 零改动）；训练中可经 config pm_sk_t_max 调小（如 10）——实测
        # 谱范数仍=1.0（信号守恒红线不破）、双随机偏差 ≤1.3e-2，吞吐 ×1.7（无 .item() 同步）。
        self.t_max = t_max
        self.constrain = constrain
        n = n_stream
        # 动态投影 φ：0 初始化 ⇒ 初始化态动态路径无贡献（HC §2.3）
        self.phi = nn.Parameter(torch.zeros(n * d_model, n * n + 2 * n))
        # 静态偏置 b：b_pre = logit(1/n)（σ 反函数），b_post = b_res = 0（恒等初始化）
        b = torch.zeros(n * n + 2 * n)
        b[:n] = math.log(1.0 / n / (1.0 - 1.0 / n))  # = -ln(n-1)
        self.bias = nn.Parameter(b)
        # 门控因子 α：init 0.01（mHC 附录 A.1）
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

    def forward(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """S [B,T,n,d] → (H_pre [B,T,n], H_post [B,T,n], H_res [B,T,n,n])，均 fp32。"""
        B, T, n, d = S.shape
        # 混合系数全部在 fp32 内计算（对齐原文 Eq.10–19 的 fp32 系数精度，plan 红线）
        with torch.autocast(device_type=S.device.type, enabled=False):
            v = S.float().reshape(B, T, n * d)  # vec(S)，流维在前（Eq.7 的 flatten）
            # Eq.7：RMSNorm(x⃗)，对 n·d 维归一（原文将 norm 权重吸收进 φ，Eq.14–16）
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
            dyn = v @ self.phi  # [B,T,n²+2n]
            d_pre, d_post, d_res = dyn.split([n, n, n * n], dim=-1)
            b_pre, b_post, b_res = self.bias.split([n, n, n * n], dim=-1)
            h_pre = self.alpha_pre * d_pre + b_pre  # Eq.7
            h_post = self.alpha_post * d_post + b_post
            h_res = (self.alpha_res * d_res + b_res).reshape(B, T, n, n)
            if self.constrain:
                # Eq.8：σ 非负约束（H_pre）、2σ 非负约束（H_post）、Birkhoff 投影（H_res）
                h_pre = torch.sigmoid(h_pre)
                h_post = 2 * torch.sigmoid(h_post)
                h_res = sinkhorn_knopp(h_res, self.t_max)
            # else：无约束 HC 消融对照（非默认、非原文 mHC），验证 Sinkhorn 的稳定作用
        return h_pre, h_post, h_res

    def read(self, S: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """读：聚合 n 流 → 子层输入 u = H_pre·S（Eq.3 的 H_pre·x_l）。

        精度（吞吐优化 P0，2026-07-30）：累加由 fp64 改为 **fp32**——fp64 在消费级 GPU
        （sm_120）吞吐仅 fp32 的 ~1/32，是 PM-stream ×0.35 瓶颈主因之一。fp32 累加的
        多流归约浮点噪声 ~1e-7/token 相对，经 12 层累积仍 ≪ 恒等判据 rel<1e-5
        （tests/test_pmstream.py 已用相对容差），系数本身亦按原文在 fp32 生成。
        恒等初始化下 H_pre=(1/n)·1、各流相等，u=Σ(1/n)x 与单流 x 的 fp32 偏差
        经 final RMSNorm 吸收，相对误差保持 <1e-5（test_a_identity_init 全绿佐证）。
        einsum 优化：h_pre [B,T,n] 与 S [B,T,n,d] 归约，fp32 单次 einsum 无中间大张量。
        """
        u = torch.einsum("btn,btnd->btd", h_pre.float(), S.float())
        return u.to(S.dtype)

    def write(
        self,
        S: torch.Tensor,
        m: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """写：S ← H_res·S + H_postᵀ·m（Eq.3：残差混合 + 子层输出分配回 n 流）。fp32 累加同 read。

        einsum 优化：两项合并前先算 H_res·S（[B,T,n,n]×[B,T,n,d]→[B,T,n,d]），
        再加 H_postᵀ·m 外积广播，fp32 单次 pass，避免 fp64 的 32× 吞吐惩罚。
        """
        out = torch.einsum("btjk,btkd->btjd", h_res.float(), S.float())
        out = out + h_post.float().unsqueeze(-1) * m.float().unsqueeze(2)
        return out.to(S.dtype)

    @staticmethod
    def pm_index(n_stream: int) -> int:
        """PM-stream（感知-记忆流）的流索引：末位流（设计文档 §13.4）。"""
        return n_stream - 1
