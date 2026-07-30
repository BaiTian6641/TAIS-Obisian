"""优化器包：Muon（arXiv:2412.02684 谱系）+ 内部 AdamW 分组。"""
from .muon import Muon, build_muon_optimizer, zeropower_via_newtonschulz5

__all__ = ["Muon", "build_muon_optimizer", "zeropower_via_newtonschulz5"]
