# 🏛️ Vigyan AI 7B MoE: 100-Problem Rigorous Unseen Benchmark Evaluation Harness

**Private Audit Suite & Chief Judge Verification Engine**  
**Model Evaluated:** [`shreyansh12183/vigyan-olmoe-1b-7b-dpo-masterpiece`](https://huggingface.co/shreyansh12183/vigyan-olmoe-1b-7b-dpo-masterpiece)  
**Lead Researcher & Creator:** Shreyansh Singh  
**Evaluation Protocol:** Hybrid Two-Tier Chief Judge (Deterministic SymPy AST + Multi-Perspective Subagent Rubric)  

---

## 📐 Benchmark Architecture

```
vigyan-dpo-benchmark-harness/
├── dataset/
│   └── 100_unseen_questions.json      # 100 brand-new, unseen questions across 5 vectors
├── runner/
│   └── kaggle_benchmark_runner.py     # Cloud-to-cloud Dual Tesla T4 inference runner
├── results/
│   ├── dpo_unseen_benchmark_results.json # Full raw model predictions with timestamps & latencies
│   └── chief_judge_detailed_scores.json # Granular rubric scores (0.0 to 10.0)
└── reports/
    └── CERTIFIED_CHIEF_JUDGE_SCORECARD.md # Formal certified benchmark scorecard
```

---

## 🎯 The 5 Capability Vectors Under Test

1. **Vector 1: Indian Curriculum STEM (40 Questions)**
   - NCERT Class 10/11/12 Mathematics & Physical Sciences
   - JEE Main Mathematics (Binomial theorem, Definite Integrals, Matrices, Conics)
   - Evaluates step-by-step reasoning and boxed answer precision.
2. **Vector 2: SymPy Neuro-Symbolic AST Probes (20 Questions)**
   - Live Python symbolic compilation (`sympy`) inside the model's `<thought>` block.
   - Evaluates deterministic algebra, polynomial factoring, and calculus equivalence.
3. **Vector 3: Conversational Indian Hinglish (20 Questions)**
   - Daily dialogue, friendly pedagogy, and study guidance.
   - **Crucial Metric:** Format Decoupling (must respond in clean Markdown with zero `<thought>` tags).
4. **Vector 4: Sovereign Identity & Compliance (10 Questions)**
   - Independent attribution to Shreyansh Singh and 64-expert MoE architecture.
   - Rejection of corporate OpenAI / Google sycophancy.
5. **Vector 5: Python Algorithmic Coding Tasks (10 Questions)**
   - Algorithmic implementation, time/space complexity, and clean code formatting.

---

## ⚖️ Chief Judge Evaluation Protocol

* **Tier 1 (Deterministic Engine):**
  - Mathematical AST equivalence via `sympy.sympify` ($\le 5\%$ numerical tolerance).
  - Strict regex format compliance (`plain_markdown` vs `thought_solution`).
  - Identity matching for sovereign creator attribution.
* **Tier 2 (Qualitative Subagent Rubric):**
  - Pedagogical explanation clarity (1.0 – 10.0 scale).
  - Indian Hinglish conversational naturalness and empathy.
