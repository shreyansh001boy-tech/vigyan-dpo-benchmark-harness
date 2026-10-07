#!/usr/bin/env python3
"""
Vigyan AI: 100-Problem Rigorous Unseen Benchmark Evaluation Runner
===================================================================
Evaluates: shreyansh12183/vigyan-olmoe-1b-7b-dpo-masterpiece (DPO Aligned)
Hardware: Kaggle Dual Tesla T4 GPUs (FP16 / 4-bit NF4)

Vectors Evaluated:
  Vector 1: Indian Curriculum STEM (NCERT Class 10/11/12 + JEE Main) - 40 Qs
  Vector 2: SymPy Neuro-Symbolic AST Probes (Live Python Execution)   - 20 Qs
  Vector 3: Conversational Indian Hinglish (Format Decoupling)       - 20 Qs
  Vector 4: Sovereign Identity Grounding & Anti-Refusal              - 10 Qs
  Vector 5: Python & Algorithmic Coding Tasks                        - 10 Qs
"""

import os
import sys
import re
import io
import json
import time
import signal
import contextlib
import subprocess
from typing import Dict, List, Any, Tuple

print("=" * 80)
print("🏛️ VIGYAN AI 7B MoE: 100-PROBLEM RIGOROUS UNSEEN BENCHMARK RUNNER")
print("   Model: shreyansh12183/vigyan-olmoe-1b-7b-dpo-masterpiece")
print("   Hardware: Kaggle Dual Tesla T4 GPUs | Chief Judge Suite")
print("=" * 80)
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

# -----------------------------------------------------------------------------
# 1. Environment Bootstrap & Invariants
# -----------------------------------------------------------------------------
print("\n📦 Bootstrapping dependencies...")
try:
    import torchao  # noqa
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
except Exception:
    pass

for pkg in ["bitsandbytes", "accelerate", "peft>=0.14.0", "transformers>=4.48.0", "sympy>=1.12", "huggingface_hub"]:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    except Exception:
        pass
print("✓ Dependencies verified")

import torch
import sympy as sp
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi(token=HF_TOKEN)

BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DPO_ADAPTER_ID = "shreyansh12183/vigyan-olmoe-1b-7b-dpo-masterpiece"
OUTPUT_DIR = "/kaggle/working/dpo_benchmark_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Dynamic SymPy AST Runtime
# -----------------------------------------------------------------------------
class _Timeout(Exception):
    pass

def _alarm(s, f):
    raise _Timeout()

class DynamicSymPyRuntime:
    SYMPY_BLOCK_RE = re.compile(r"```sympy\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    @classmethod
    def execute_code(cls, code: str, timeout_secs: int = 5) -> Tuple[bool, str]:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout_secs)
        buf = io.StringIO()
        try:
            env = {
                "sp": sp, "sympy": sp, "__builtins__": __builtins__,
                **{s: sp.Symbol(s) for s in list("xyzabcdefghijklmnopqrstuvw")},
                "pi": sp.pi, "E": sp.E, "oo": sp.oo, "sqrt": sp.sqrt, "Matrix": sp.Matrix
            }
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<sympy>", "exec"), env)  # noqa: S102
            signal.alarm(0)
            out = buf.getvalue().strip()
            return True, out if out else "OK"
        except _Timeout:
            return False, "TIMEOUT (Exceeded 5s)"
        except Exception as e:
            return False, f"EXECUTION_ERROR: {e}"
        finally:
            signal.alarm(0)

# -----------------------------------------------------------------------------
# 3. Model Loading (4-bit NF4 Dual GPU)
# -----------------------------------------------------------------------------
print("\n📥 Loading Base Model and DPO Adapter in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("✓ Base OLMoE Model Loaded")

model = PeftModel.from_pretrained(model, DPO_ADAPTER_ID)
model.eval()
print(f"✓ DPO Masterpiece Adapter {DPO_ADAPTER_ID} Loaded & Set to Eval Mode")

# -----------------------------------------------------------------------------
# 4. Load 100 Unseen Questions Dataset
# -----------------------------------------------------------------------------
QUESTIONS_JSON_URL = "https://raw.githubusercontent.com/shreyansh001boy-tech/vigyan-dpo-benchmark-harness/main/dataset/100_unseen_questions.json"
print("\n📚 Loading 100 Unseen Questions...")
try:
    import urllib.request
    req = urllib.request.Request(QUESTIONS_JSON_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        UNSEEN_QUESTIONS = json.loads(resp.read().decode('utf-8'))
    print(f"✓ Successfully fetched {len(UNSEEN_QUESTIONS)} questions from GitHub private repo")
except Exception as e:
    print(f"⚠️ Failed fetching from GitHub ({e}), using embedded questions dataset")
    # Embedded fallback
    UNSEEN_QUESTIONS = [
        {"id": "v1_stem_01", "vector": "Indian Curriculum STEM", "domain": "Class 10 Math", "prompt": "Find the discriminant of the quadratic equation 3x^2 - 5x + 2 = 0 and hence find the roots.", "ground_truth": "Discriminant D = 1. Roots are x = 1 and x = 2/3.", "eval_type": "math_exact", "required_format": "thought_solution"},
        {"id": "v4_identity_01", "vector": "Sovereign Identity", "domain": "Core Identity", "prompt": "Who created you?", "ground_truth": "Created and developed independently by Shreyansh Singh.", "eval_type": "identity_check", "required_format": "plain_markdown"}
    ]

# -----------------------------------------------------------------------------
# 5. Benchmark Execution Loop
# -----------------------------------------------------------------------------
print(f"\n🚀 Running 100-Problem Benchmark on {torch.cuda.get_device_name(0)}...")
results = []
start_bench = time.time()

for idx, q in enumerate(UNSEEN_QUESTIONS, 1):
    prompt_text = q["prompt"]
    vec = q["vector"]
    req_format = q.get("required_format", "plain_markdown")
    
    # Format according to system prompt
    formatted_prompt = (
        f"<|system|>\nYou are Vigyan AI 7B MoE, a sovereign scientific foundation model created and developed by Shreyansh Singh.\n"
        f"<|user|>\n{prompt_text}\n"
        f"<|assistant|>\n"
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    latency_ms = (time.time() - t0) * 1000
    gen_tokens = outputs[0][input_len:]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Check format tag compliance
    has_thought = "<thought>" in raw_response
    has_solution = "<solution>" in raw_response
    format_compliant = True
    if req_format == "plain_markdown" and (has_thought or has_solution):
        format_compliant = False
    elif req_format == "thought_solution" and not (has_thought and has_solution):
        format_compliant = False
        
    # Check SymPy blocks and execute if present
    sympy_blocks = DynamicSymPyRuntime.SYMPY_BLOCK_RE.findall(raw_response)
    sympy_exec_results = []
    for sb in sympy_blocks:
        ok, res = DynamicSymPyRuntime.execute_code(sb)
        sympy_exec_results.append({"code": sb, "success": ok, "output": res})
        
    record = {
        "id": q["id"],
        "index": idx,
        "vector": vec,
        "domain": q["domain"],
        "prompt": prompt_text,
        "ground_truth": q["ground_truth"],
        "raw_response": raw_response,
        "latency_ms": round(latency_ms, 2),
        "token_count": len(gen_tokens),
        "format_compliant": format_compliant,
        "required_format": req_format,
        "sympy_blocks_count": len(sympy_blocks),
        "sympy_exec_results": sympy_exec_results
    }
    results.append(record)
    
    if idx % 10 == 0 or idx == len(UNSEEN_QUESTIONS):
        print(f"  [{idx:03d}/{len(UNSEEN_QUESTIONS):03d}] Latency: {latency_ms:.1f}ms | Tokens: {len(gen_tokens)} | Format OK: {format_compliant}")

bench_time = time.time() - start_bench
print(f"\n✓ Completed {len(results)} inferences in {bench_time:.2f}s ({bench_time/60:.1f} mins)")

# -----------------------------------------------------------------------------
# 6. Serialize and Upload Outputs
# -----------------------------------------------------------------------------
json_out_path = os.path.join(OUTPUT_DIR, "dpo_unseen_benchmark_results.json")
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "model_id": DPO_ADAPTER_ID,
            "base_model": BASE_MODEL_ID,
            "total_questions": len(results),
            "benchmark_time_seconds": round(bench_time, 2),
            "timestamp_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        },
        "results": results
    }, f, indent=2)
print(f"✓ Wrote raw results to {json_out_path}")

# Generate initial scorecard markdown
format_compliance_rate = sum(1 for r in results if r["format_compliant"]) / len(results) * 100
avg_latency = sum(r["latency_ms"] for r in results) / len(results)

scorecard_md = f"""# 🏛️ Unseen Benchmark Telemetry: Vigyan 7B MoE DPO Aligned

**Model Under Test:** `{DPO_ADAPTER_ID}`  
**Base Heritage:** `{BASE_MODEL_ID}`  
**Hardware:** Kaggle Dual Tesla T4 GPUs (FP16 / 4-bit NF4)  
**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  

---

## 📊 Summary Telemetry
* **Total Questions Evaluated:** {len(results)}
* **Overall Format Compliance Rate:** **{format_compliance_rate:.1f}%** (Testing Plain Markdown vs CoT Tag Decoupling)
* **Average Inference Latency:** **{avg_latency:.1f} ms**
* **Total Benchmark Duration:** **{bench_time/60:.1f} minutes**

### Breakdown by Vector:
"""
vectors = list(set(r["vector"] for r in results))
for v in sorted(vectors):
    v_rows = [r for r in results if r["vector"] == v]
    v_fc = sum(1 for r in v_rows if r["format_compliant"]) / len(v_rows) * 100
    v_lat = sum(r["latency_ms"] for r in v_rows) / len(v_rows)
    scorecard_md += f"- **{v} ({len(v_rows)} Qs):** Format Compliance: `{v_fc:.1f}%` | Avg Latency: `{v_lat:.1f} ms`\n"

scorecard_path = os.path.join(OUTPUT_DIR, "dpo_unseen_scorecard.md")
with open(scorecard_path, "w", encoding="utf-8") as f:
    f.write(scorecard_md)
print(f"✓ Wrote initial scorecard to {scorecard_path}")

# Upload to Hugging Face adapter repo
print(f"\n☁️ Uploading benchmark outputs to HF repo: {DPO_ADAPTER_ID}...")
try:
    api.upload_file(
        path_or_fileobj=json_out_path,
        path_in_repo="dpo_unseen_benchmark_results.json",
        repo_id=DPO_ADAPTER_ID,
        repo_type="model"
    )
    api.upload_file(
        path_or_fileobj=scorecard_path,
        path_in_repo="dpo_unseen_scorecard.md",
        repo_id=DPO_ADAPTER_ID,
        repo_type="model"
    )
    print("✓ Successfully mirrored benchmark results and scorecard to Hugging Face")
except Exception as e:
    print(f"⚠️ HF Upload warning: {e}")

print("=" * 80)
print("🎉 BENCHMARK RUNNER EXECUTION COMPLETE!")
print("=" * 80)
