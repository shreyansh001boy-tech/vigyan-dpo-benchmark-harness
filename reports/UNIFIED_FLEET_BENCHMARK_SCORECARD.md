# 🏛️ Vigyan AI: Unified Fleet Benchmark & Sovereign Verification Audit

> **Independent Benchmark Verification**: Certified tamper-proof scorecard for the Vigyan model series (**1.5B, 3B, 7B, 32B**).
> All raw model prediction JSON logs, deterministic Chief Judge evaluation scripts, and prompt vectors are published openly in this repository.

## 1. Sovereign Fleet Benchmark Matrix

| Fleet Tier | Parameters | Base Scaffold | Exam Suite | Total / Passed | Pass Rate (%) | Core Breakthrough |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Vigyan-1.5B** | 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 100 Frontier Exam | 100 / 43 | **43.0%** | High-density edge reasoning |
| **Vigyan-3B** | 3B | `Qwen/Qwen2.5-3B-Instruct` | 100 Frontier Exam | 100 / 69 | **69.0%** | 🚀 **Surpassed 7B!** (+4.0% overall, +10.0% RTL) |
| **Vigyan-7B** | 7B | `allenai/OLMo-2-1124-7B-Instruct` | 100 Frontier Exam | 100 / 65 | **65.0%** | Foundational STEM anchor |
| **Vigyan-32B Titan** | 32.5B | `allenai/OLMo-2-32B` | GSM8K / Gate 1 Probe | 1319 / 1110 | **84.2%** | QK-Norm 15th Shard Restored (AWS SageMaker `ml.g5.16xlarge`) |

---

## 2. Quad-Master 100-Question Frontier Domain Breakdown

| Capability Vector | Vigyan-1.5B | Vigyan-3B 🚀 | Vigyan-7B | Best In Class |
| :--- | :---: | :---: | :---: | :---: |
| **Silicon RTL & EDA** | 30.0% | **85.0%** | 75.0% | 85.0% (Vigyan-3B) |
| **PhD Pure Mathematics** | 75.0% | **85.0%** | 85.0% | 85.0% (Vigyan-3B & 7B Tie) |
| **Astrophysics & Orbital Mechanics** | 45.0% | **65.0%** | 55.0% | 65.0% (Vigyan-3B) |
| **Biomedical & Quantum Chemistry** | 40.0% | **60.0%** | 65.0% | 65.0% (Vigyan-7B) |
| **Indian STEM & Hinglish** | 25.0% | **50.0%** | 45.0% | 50.0% (Vigyan-3B) |

---

## 3. Engineering Breakthrough: Why Vigyan-3B Surpassed 7B

1. **Modern Foundation Scaffold**: Vigyan-3B is initialized from `Qwen/Qwen2.5-3B-Instruct`, featuring RoPE 128k context windows, SwiGLU activations, and dense mathematical pre-training, outclassing older 7B architectures.
2. **DARE-TIES High-Density Merging**: A density parameter of `0.75` (25% drop rate) retained high-frequency Verilog syntax and chemical IUPAC tokens that are typically wiped during aggressive pruning.
3. **Token Horizon Protection**: Generation limit expanded to `512` tokens, eliminating truncation artifacts where DeepSeek-style reasoning scratchpads previously ran out of budget before emitting `endmodule`.

---

## 4. 1-Click Peer Reproduction Instructions

```bash
# 1. Clone repository
git clone https://github.com/shreyansh001boy-tech/vigyan-dpo-benchmark-harness.git
cd vigyan-dpo-benchmark-harness

# 2. Run instant deterministic verification (SymPy CAS + Verilog AST)
python reproduce_fleet_audit.py --model all
```

*Generated automatically by `reproduce_fleet_audit.py` on behalf of Vigyan AI.*
