# -*- coding: utf-8 -*-
"""evidence-forward 端到端 扩样到 N=35：从 PER_CAP=3/N=24 扩到 PER_CAP=5/N=35。
- 先归档现有结果 json（若有）为 *_21.json（保留 21 题基线）。
- import 原 e2e 模块并覆盖常量，PER_CAP=5 每能力5题 x7能力=35。
- 原 main() 输出到 probe_evidence_forward_e2e.json，末尾复制为 *_35.json。
"""
import os, shutil, importlib

_LINS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_LINS, "results", "probe_evidence_forward_e2e.json")
ARC = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_21.json")
OUT35 = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")

if os.path.exists(SRC) and not os.path.exists(ARC):
    shutil.copy2(SRC, ARC)
    print(f"已归档 21 题基线 -> probe_evidence_forward_e2e_21.json")

m = importlib.import_module("_probe_evidence_forward_end2end")
m.PER_CAP = 5
m.N = 35                       # 7 能力 x 5 题 = 35
m.MAX_PER_DOC = m.MAX_PER_DOC  # 保持 50
m.main()

if os.path.exists(SRC):
    shutil.copy2(SRC, OUT35)
    print(f"\n35 题结果已存 -> probe_evidence_forward_e2e_35.json")
