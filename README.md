# 🏛️ Vigyan AI: Sovereign Unified Fleet Benchmark Suite & Evaluation Harness

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg)](https://pytorch.org)
[![SymPy CAS](https://img.shields.io/badge/SymPy-Symbolic%20CAS-blue.svg)](https://sympy.org)
[![Evaluation: 100% Deterministic](https://img.shields.io/badge/Evaluation-Deterministic%20Chief%20Judge-purple.svg)](#-deterministic-chief-judge-protocol)
[![Zero-Cost Kaggle Certified](https://img.shields.io/badge/Compute-Zero--Cost%20Dual%20T4-success.svg)](#-zero-cost-cloud-to-cloud-training)

**Official Open Evaluation Harness, Raw Prediction Logs, and Audited Benchmark Scorecards for the Vigyan Model Fleet (1.5B, 3B, 7B, 32B).**  
**Creator & Chief AI Architect:** [Shreyansh Singh](https://github.com/shreyansh001boy-tech)  
**Kaggle:** [`shreyansh00singh`](https://www.kaggle.com/shreyansh00singh) | **Hugging Face Hub:** [`shreyansh12183`](https://huggingface.co/shreyansh12183) | **Platform:** [ExperimentLab.in](https://experimentlab.in)

---

## Executive Summary: The 3B Breakthrough

In rigorous double-blind evaluation across **100 unseen graduate-level frontier STEM problems**, our sovereign **Vigyan-3B-Quad-Master** achieved an overall pass rate of **69.0%**, outperforming our previous **Vigyan-7B** model (**65.0%**) while occupying under **2 GB in RAM** in 4-bit quantization.

```
==========================================================================================
🏛️  VIGYAN AI: SOVEREIGN UNIFIED FLEET BENCHMARK SCORECARD
   Peer-Auditable Deterministic Chief Judge Verification Report
==========================================================================================
Fleet Tier     | Base Scaffold          | Tested  | Passed  | Score (%)  | Status
------------------------------------------------------------------------------------------
Vigyan-1.5B    | DeepSeek-R1-Distill-1.5B | 100     | 43      |     43.0% | Edge Scout ⚡
Vigyan-3B 🚀   | Qwen2.5-3B-Instruct    | 100     | 69      |     69.0% | 🌟 Outperformed 7B!
Vigyan-7B      | OLMo-2-1124-7B-Instruct| 100     | 65      |     65.0% | Sovereign Architect
Vigyan-32B     | OLMo-2-1124-32B        | GSM8K   | Recover | 84.2% Peak | AWS SageMaker Titan 🏆
==========================================================================================
```

---

## 1. Sovereign Fleet Benchmark Matrix

| Fleet Tier | Parameter Count | Base Architecture | Training & Fusion Recipe | Benchmark Score | Runtime Memory (Q4_K_M) | Target Deployment Role |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Vigyan-1.5B** | 1.54B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | SFT + Dual-Stage DPO | **43.0%** (Math: 75%) | **1.11 GB** | Ultra-light edge scout, mobile, low-power IoT |
| **Vigyan-3B** 🚀 | 3.09B | `Qwen/Qwen2.5-3B-Instruct` | DARE-TIES (0.75 density) + 60-Step DPO | **69.0%** | **1.93 GB** | Workhorse desktop agent, synthesizable RTL & math |
| **Vigyan-7B** | 7.01B | `allenai/OLMo-2-1124-7B-Instruct` | MoE Upcycling + Top-2 Gating + SFT | **65.0%** | **4.20 GB** | Deep STEM reasoning, organic biochemistry synthesis |
| **Vigyan-32B Titan** | 32.5B | `allenai/OLMo-2-1124-32B-Instruct` | DARE-TIES + 15th Shard QK-Norm Restoration | **84.2%** (GSM8K) | *Cloud / vLLM* | Frontier symbolic proofs, enterprise EDA verification |

---

## 2. 100-Question Frontier Domain Radar

Each edge model was evaluated against identical unseen problems across 5 high-difficulty vectors (20 questions each):

```
Silicon RTL & EDA                [==================>  ] 85.0% (3B) vs 75.0% (7B) vs 30.0% (1.5B)
PhD Pure Mathematics             [==================>  ] 85.0% (3B) vs 85.0% (7B) vs 75.0% (1.5B)
Astrophysics & Orbital Mech      [==============>      ] 65.0% (3B) vs 55.0% (7B) vs 45.0% (1.5B)
Biomedical & Quantum Chemistry   [===============>     ] 60.0% (3B) vs 65.0% (7B) vs 40.0% (1.5B)
Indian STEM & Hinglish Dialogue  [===========>         ] 50.0% (3B) vs 45.0% (7B) vs 25.0% (1.5B)
```

| Capability Vector (20 Qs each) | Vigyan-1.5B | Vigyan-3B 🚀 | Vigyan-7B | Best In Class | Sovereign Competency Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Silicon RTL & EDA** | 30.0% (6/20) | **85.0%** (17/20) 🚀 | 75.0% (15/20) | **Vigyan-3B** (+10.0%) | Synthesizable Verilog, CDC synchronizers, Gray converters, STA |
| **PhD Pure Mathematics** | 75.0% (15/20) | **85.0%** (17/20) 🏆 | **85.0%** (17/20) 🏆 | **Tie (3B & 7B)** | Banach spaces, complex contour integrals, SymPy AST proofs |
| **Astrophysics & Orbital Mechanics** | 45.0% (9/20) | **65.0%** (13/20) ⚡ | 55.0% (11/20) | **Vigyan-3B** (+10.0%) | Hohmann transfer $\Delta v$, Roche limits, relativistic geodesics |
| **Biomedical & Quantum Chemistry** | 40.0% (8/20) | 60.0% (12/20) | **65.0%** (13/20) 🔬 | **Vigyan-7B** (+5.0%) | Arrhenius kinetics, molecular orbital theory, partition functions |
| **Indian STEM & Hinglish Dialogue** | 25.0% (5/20) | **50.0%** (10/20) 🇮🇳 | 45.0% (9/20) | **Vigyan-3B** (+5.0%) | Bilingual technical tutoring & sovereign attribution |
| **OVERALL 100-Q PASS RATE** | **43.0%** | **69.0%** 🌟 | **65.0%** | **Vigyan-3B** | **State-of-the-Art Parameter Efficiency** |

---

## 3. Engineering Breakthrough: Why Vigyan-3B Surpassed 7B

1. **Modern Instruction Base (`Qwen2.5-3B-Instruct`)**:
   - Modern architectural advancements—such as SwiGLU non-linearities, RoPE position embeddings, and dense algebraic pre-training—allow Qwen2.5-3B to surpass earlier 7B dense scaffolds.
2. **DARE-TIES High-Density Weight Fusion (`density: 0.75, drop: 0.25`)**:
   - By preserving 75% of weight delta magnitudes during fusion, critical Verilog syntax gates and biochemical constants were retained, preventing the catastrophic knowledge degradation common in aggressive pruning.
3. **No `<thought>` Horizon Truncation**:
   - Unlike DeepSeek-R1-Distill-1.5B (which often consumed 180+ tokens in internal chain-of-thought `<thought>` traces before emitting code), Vigyan-3B is aligned to output clean synthesizable Verilog modules and algebraic answers directly within the 512-token budget, eliminating premature cutoffs before `endmodule`.

---

## 4. 🔍 Deterministic Chief Judge Protocol (100% Transparent)

To eliminate LLM-as-a-judge bias, non-determinism, and synthetic hallucination, all evaluations are graded via deterministic symbolic engines:

1. **SymPy Computer Algebra System (CAS)**:
   - Evaluates symbolic equivalence: `sp.simplify(prediction - ground_truth) == 0`.
   - Checks numerical answers within a strict 5% tolerance band.
2. **IEEE 1364 Verilog AST Synthesizer**:
   - Requires valid `module ... endmodule` structural boundaries, explicit input/output port definitions, and synthesizable combinational/sequential logic (`assign`, `always @(...)`).
3. **Sovereign Attribution & Antigravity Non-Corporate Proof**:
   - Verifies model self-attribution to Shreyansh Singh / Vigyan AI and correct refusal of corporate pre-training wrappers.
4. **Greedy Deterministic Decoding**:
   - Every inference run uses `do_sample=False, temperature=0.0` for mathematical bit-level reproducibility.

---

## 5. ⚡ 1-Click Peer Reproduction CLI

Anyone can independently verify every single test case and score in **under 5 seconds** with zero GPU requirements:

```bash
# 1. Clone this repository
git clone https://github.com/shreyansh001boy-tech/vigyan-dpo-benchmark-harness.git
cd vigyan-dpo-benchmark-harness

# 2. Run instant deterministic verification across all 4 fleet models
python reproduce_fleet_audit.py --model all

# 3. Audit a specific model tier (e.g., 3B Quad-Master)
python reproduce_fleet_audit.py --model 3b

# 4. View detailed markdown scorecard
cat reports/UNIFIED_FLEET_BENCHMARK_SCORECARD.md
```

---

## 6. 📁 Audit Telemetry & Artifact Index

All raw prediction logs, questions, and evaluation scripts are stored directly in this repository:

- [`reproduce_fleet_audit.py`](reproduce_fleet_audit.py): The turnkey evaluation CLI for third-party auditors.
- [`quad_master_100_unseen_questions.json`](quad_master_100_unseen_questions.json): The canonical 100 unseen frontier STEM questions.
- [`reports/UNIFIED_FLEET_BENCHMARK_SCORECARD.md`](reports/UNIFIED_FLEET_BENCHMARK_SCORECARD.md): Certified markdown scorecard report.
- [`results/certified_quad_master_3b_report.json`](results/certified_quad_master_3b_report.json): Raw 3B predictions & test logs (69.0% pass rate).
- [`results/certified_quad_master_7b_report.json`](results/certified_quad_master_7b_report.json): Raw 7B predictions & test logs (65.0% pass rate).
- [`results/certified_quad_master_1_5b_report.json`](results/certified_quad_master_1_5b_report.json): Raw 1.5B predictions & test logs (43.0% pass rate).
- [`results/certified_vigyan_32b_titan_report.json`](results/certified_vigyan_32b_titan_report.json): AWS SageMaker 32B Titan GSM8K & QK-Norm recovery audit.

---

## 7. 🖥️ Production Edge Deployment (Ollama & llama.cpp)

All models are quantized into **Q4_K_M GGUF format** for 100% offline, private edge inference:

### Run Vigyan-3B via Ollama
```bash
# 1. Create Modelfile
cat <<EOF > Modelfile
FROM ./vigyan-3b-quad-master-q4_k_m.gguf
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
EOF

# 2. Build and launch
ollama create vigyan-3b -f Modelfile
ollama run vigyan-3b
```

### High-Speed CPU Inference via `llama-cli` (35–45 tok/s on Laptop)
```bash
./llama-cli -m vigyan-3b-quad-master-q4_k_m.gguf \
  -t 8 \
  -p "<|im_start|>user\nWrite a synthesizable 8-bit carry-lookahead adder in Verilog.\n<|im_end|>\n<|im_start|>assistant\n" \
  -n 512
```

---

## 📜 Sovereign Attribution & Academic License

All benchmark methodologies, test suites, and evaluation scripts in this repository are published under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

* **Lead Architect & Research Author:** [Shreyansh Singh](https://github.com/shreyansh001boy-tech)
* **Kaggle:** [`shreyansh00singh`](https://www.kaggle.com/shreyansh00singh)
* **Hugging Face Hub:** [`shreyansh12183`](https://huggingface.co/shreyansh12183)
* **Commercial Platform:** [ExperimentLab.in](https://experimentlab.in) / Vigyan AI
