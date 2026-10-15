# 🏛️ Vigyan AI: Sovereign Quad-Master Benchmark Suite & Trinity Scorecard

**Empirical Evaluation Harness for Sovereign STEM Small Language Models (SLMs) & MoEs**  
**Creator & Lead Architect:** [Shreyansh Singh](https://github.com/shreyansh001boy-tech)  
**Kaggle Profile:** [`shreyansh00singh`](https://www.kaggle.com/shreyansh00singh) | **Hugging Face Hub:** [`shreyansh12183`](https://huggingface.co/shreyansh12183)  
**Evaluation Protocol:** 100 Unseen Frontier STEM Questions evaluated via Chief Judge SymPy AST Engine + Hardware Verilog Synthesizability Rubric.

---

## 🏆 The Trinity Breakthrough: 3B Quad-Master Outperforms 7B

Through high-density **DARE-TIES Weight Fusion (density: 0.75, drop: 0.25)** and targeted **60-step DPO preference alignment**, our sovereign **Vigyan-3B-Quad-Master** has officially surpassed our earlier 7B model on graduate-level STEM benchmarks, achieving **69.0% overall pass rate** while remaining under **2 GB in RAM**!

```
========================================================================================
                  VIGYAN AI QUAD-MASTER TRINITY BENCHMARK VERDICT
========================================================================================
  • Vigyan 3B Quad-Master (1.93 GB GGUF):   69.0% Pass Rate (69 / 100 Passed) 🌟 SOTA
  • Vigyan 7B Quad-Master (4.20 GB GGUF):   65.0% Pass Rate (65 / 100 Passed)
  • Vigyan 1.5B Quad-Master (1.11 GB GGUF): 43.0% Pass Rate (Math: 75%, RTL: 30%)
  • Vigyan 32B Titan (AWS SageMaker):       84.2% GSM8K Arithmetic Mastery
========================================================================================
```

---

## 📊 Comprehensive 100-Question Frontier STEM Scorecard

All models were evaluated dynamically against the exact same **100 unseen, graduate-level frontier STEM questions** across 5 distinct capability vectors:

| Capability Vector (20 Qs each) | 1.5B Quad-Master (DeepSeek-R1 Distill) | 3B Quad-Master (Qwen2.5-Instruct) | 7B Quad-Master (OLMo-2-7B) | Sovereign Competency Focus |
| :--- | :---: | :---: | :---: | :--- |
| **Silicon RTL & EDA** | 30.0% (6/20) | **85.0%** (17/20) 🚀 | 75.0% (15/20) | Synthesizable Verilog, clock gating, setup/hold slack |
| **PhD Pure Mathematics** | 75.0% (15/20) | **85.0%** (17/20) 🏆 | **85.0%** (17/20) 🏆 | Complex integrals, Banach spaces, SymPy AST derivations |
| **Astrophysics & Orbital Mechanics**| 45.0% (9/20) | **65.0%** (13/20) ⚡ | 55.0% (11/20) | Hohmann transfers, Roche limit, orbital velocities |
| **Biomedical & Quantum Chemistry**  | 40.0% (8/20) | **60.0%** (12/20) 🔬 | **65.0%** (13/20) | Reaction kinetics, molecular orbitals, partition functions |
| **Indian STEM & Hinglish Dialogue** | 25.0% (5/20) | **50.0%** (10/20) 🇮🇳 | 45.0% (9/20) | Bilingual technical tutoring & sovereign identity |
| **OVERALL 100-Q PASS RATE** | **43.0%** | **69.0%** 🌟 | **65.0%** | **State-of-the-Art Parameter Efficiency** |
| **GGUF Q4_K_M Binary Size** | **1.11 GB** | **1.93 GB** | **4.20 GB** | **100% Offline Execution on Standard Laptops** |
| **Desktop Swarm Architecture** | **System 1 (Scout)** | **Solo Workhorse** | **System 2 (Architect)** | Fast Triage vs. Deep Synthesizer |

---

## 🔬 Architectural Post-Mortem: Why 3B Outperformed 7B

1. **Foundation Model Quality**:
   - The 3B model is built on `Qwen/Qwen2.5-3B-Instruct`, a modern instruction-tuned base with dense coding and algebraic reasoning capabilities that punch above older 7B architectures.
2. **DARE-TIES Density Boost (0.75 vs 0.60)**:
   - High-density retention (0.75) with gentle pruning (0.25) prevented syntax-critical Verilog markers and biochemical reaction constants from being zeroed out during fusion.
3. **No `<thought>` Truncation Waste**:
   - Unlike DeepSeek-R1 (which burned 180 tokens on internal thought loops before code could output), Qwen2.5 outputs synthesizable code directly. Combined with our expanded **512 token budget**, complete modules with `endmodule` finished without cutoffs.

---

## 🖥️ Production GGUF Deployment Guide (Ollama & Desktop)

All models are quantized into **Q4_K_M GGUF format** for 100% air-gapped, offline execution inside **Vigyan Pro Desktop** or standalone `llama.cpp`:

### 1. Run 3B Quad-Master via Ollama
```bash
# Create local Modelfile
cat <<EOF > Modelfile
FROM ./vigyan-3b-quad-master-q4_k_m.gguf
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
EOF

# Build & Run in terminal
ollama create vigyan-3b -f Modelfile
ollama run vigyan-3b
```

### 2. High-Speed CPU Inference via llama-cli (30–45 tok/s)
```bash
./llama-cli -m vigyan-3b-quad-master-q4_k_m.gguf \
  -t 8 \
  -p "<|im_start|>user\nWrite a synthesizable 8-bit carry-lookahead adder in Verilog.\n<|im_end|>\n<|im_start|>assistant\n" \
  -n 512
```

---

## 📁 Repository Telemetry Files

- [`results/certified_quad_master_1_5b_report.json`](results/certified_quad_master_1_5b_report.json): Raw 1.5B test audit (43.0% pass rate).
- [`results/certified_quad_master_3b_report.json`](results/certified_quad_master_3b_report.json): Raw 3B test audit (69.0% pass rate).
- [`results/certified_quad_master_benchmark_report.json`](results/certified_quad_master_benchmark_report.json): Raw 7B test audit (65.0% pass rate).
- [`quad_master_100_unseen_questions.json`](quad_master_100_unseen_questions.json): The canonical 100 unseen frontier STEM questions.

---

## 👤 Author & Research Attribution

* **Architect & Developer:** [Shreyansh Singh](https://github.com/shreyansh001boy-tech)
* **Kaggle:** [`shreyansh00singh`](https://www.kaggle.com/shreyansh00singh)
* **Hugging Face:** [`shreyansh12183`](https://huggingface.co/shreyansh12183)
* **Commercial Platform:** [ExperimentLab.in](https://experimentlab.in) / Vigyan AI
