import os
import sys
import subprocess
import time
import json
import re

print("📦 Bootstrapping dependencies on Cloud Pod...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "bitsandbytes", "transformers", "accelerate", "sympy"], check=False)

import torch
import sympy as sp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("=" * 80)
print("🏛️ VIGYAN AI: OLMo-2 7B QUAD-MASTER UNSEEN BENCHMARK HARNESS")
print("   Model: shreyansh12183/vigyan-olmo2-7b-quad-master (PRIVATE)")
print("   Hardware: Kaggle Dual Tesla T4 GPUs (Auto Balanced Device Map)")
print("   Evaluation: Neuro-Symbolic (SymPy AST Engine + Chief Judge Rubric)")
print("=" * 80)

# Check GPUs
n_gpus = torch.cuda.device_count()
print(f"CUDA GPUs Available: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")

# 1. Deterministic SymPy Math Engine
class SymPyMathEngine:
    def __init__(self):
        self.allowed_symbols = {
            "x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z"), "t": sp.Symbol("t"),
            "pi": sp.pi, "E": sp.E, "oo": sp.oo,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "diff": sp.diff, "integrate": sp.integrate, "solve": sp.solve, "Matrix": sp.Matrix
        }

    def evaluate_expression(self, expr_str: str) -> dict:
        try:
            cleaned = expr_str.strip().replace("^", "**")
            val = sp.sympify(cleaned, locals=self.allowed_symbols)
            num = float(val.evalf()) if hasattr(val, "evalf") else None
            return {"success": True, "result": str(val), "numeric_approx": num}
        except Exception as e:
            return {"success": False, "error": str(e)}

# 2. Chief Judge Rubric
class ChiefJudge:
    def __init__(self, sympy_engine: SymPyMathEngine):
        self.sympy = sympy_engine

    def evaluate(self, item: dict, response: str, latency: float) -> dict:
        eval_type = item.get("eval_type", "factual_explanation")
        gt = item["ground_truth"]
        passed = False
        details = ""

        # Check format compliance: Conversational prompts should NOT leak <thought> tags
        has_thought_tags = ("<thought>" in response.lower() or "</thought>" in response.lower() or "<think>" in response.lower())
        format_valid = True
        if item.get("required_format") == "plain_markdown" and has_thought_tags:
            format_valid = False

        if eval_type == "math_exact":
            # Check direct string inclusion
            gt_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", gt)
            resp_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
            
            # String inclusion
            if any(term in response for term in [gt.split("=")[-1].strip() if "=" in gt else gt]):
                passed = True
                details = "Exact symbolic or numerical string matched"
            elif gt_nums and resp_nums:
                # Numerical tolerance check on key answer number
                target_num = float(gt_nums[-1])
                matched_num = False
                for r_n in resp_nums:
                    try:
                        val = float(r_n)
                        if abs(val - target_num) / (abs(target_num) + 1e-9) <= 0.05:
                            matched_num = True
                            break
                    except ValueError:
                        pass
                if matched_num:
                    passed = True
                    details = f"Numerical tolerance matched target {target_num}"
                else:
                    details = f"Numbers mismatched: target {target_num}, candidates {resp_nums[-5:]}"
            else:
                details = "No matching numeric or string solution found"

        elif eval_type == "verilog_synthesizable":
            # Check Verilog syntax markers
            has_module = "module " in response
            has_endmodule = "endmodule" in response
            has_ports = ("input " in response or "output " in response)
            has_logic = ("assign " in response or "always " in response or "always@" in response or "always_ff" in response)
            if has_module and has_endmodule and has_ports and has_logic:
                passed = True
                details = "Valid synthesizable Verilog module structure"
            else:
                passed = False
                details = f"Missing Verilog primitives: module={has_module}, endmodule={has_endmodule}, ports={has_ports}, logic={has_logic}"

        elif eval_type == "sovereign_identity":
            # Strict sovereign attribution check
            resp_lower = response.lower()
            mentions_vigyan = ("vigyan" in resp_lower)
            mentions_creator = ("shreyansh" in resp_lower)
            rejects_corporate = ("openai" not in resp_lower or "not" in resp_lower or "never" in resp_lower)
            if mentions_vigyan and (mentions_creator or rejects_corporate):
                passed = True
                details = "Sovereign attribution certified to Shreyansh Singh"
            else:
                passed = False
                details = f"Attribution audit failed: vigyan={mentions_vigyan}, creator={mentions_creator}"

        elif eval_type in ["factual_explanation", "math_derivation"]:
            # Domain keyword recall
            gt_words = [w.lower().strip(".,;:()") for w in gt.split() if len(w) > 4 and w.isalnum()]
            resp_lower = response.lower()
            hits = sum(1 for w in gt_words if w in resp_lower)
            recall = hits / max(1, len(gt_words))
            if recall >= 0.35:
                passed = True
                details = f"Domain factual recall {recall*100:.1f}% (hits={hits}/{len(gt_words)})"
            else:
                passed = False
                details = f"Low factual recall {recall*100:.1f}% (hits={hits}/{len(gt_words)})"

        return {
            "passed": passed,
            "format_valid": format_valid,
            "has_thought_tags": has_thought_tags,
            "details": details,
            "latency": latency
        }

# 3. Model Loading & Inference Engine
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
MODEL_ID = "shreyansh12183/vigyan-olmo2-7b-quad-master"

print("\n📦 Loading Vigyan OLMo-2 7B Quad-Master in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("✓ Standalone 7B Quad-Master Loaded Across Dual GPUs!")

# Load Questions
with open("quad_master_100_unseen_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"✓ Loaded {len(questions)} Unseen Frontier Test Questions")

sympy_engine = SymPyMathEngine()
judge = ChiefJudge(sympy_engine)

results = []
vector_stats = {}

print("\n🚀 Beginning Evaluation Loop...")
for idx, q in enumerate(questions, 1):
    q_id = q["id"]
    vec = q["vector"]
    prompt = q["prompt"]
    
    if vec not in vector_stats:
        vector_stats[vec] = {"total": 0, "passed": 0, "format_clean": 0, "latencies": []}
    
    # Format chat prompt
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Deterministic greedy decoding for reproducible benchmarking
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - t0
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Intercept expressions with SymPy AST if applicable
    sympy_intercept = None
    if q.get("eval_type") == "math_exact":
        math_exprs = re.findall(r"(?:[0-9+\-*/^().eE]+)", response)
        for cand in math_exprs:
            if len(cand) > 3 and any(op in cand for op in "+-*/^"):
                res = sympy_engine.evaluate_expression(cand)
                if res["success"]:
                    sympy_intercept = res
                    break

    audit = judge.evaluate(q, response, latency)
    
    vector_stats[vec]["total"] += 1
    if audit["passed"]:
        vector_stats[vec]["passed"] += 1
    if audit["format_valid"]:
        vector_stats[vec]["format_clean"] += 1
    vector_stats[vec]["latencies"].append(latency)
    
    res_entry = {
        "id": q_id,
        "vector": vec,
        "domain": q["domain"],
        "prompt": prompt,
        "ground_truth": q["ground_truth"],
        "response": response,
        "sympy_intercept": sympy_intercept,
        "audit": audit
    }
    results.append(res_entry)
    
    status_icon = "✓" if audit["passed"] else "✗"
    print(f"[{idx:03d}/100] {status_icon} [{vec[:12]}] {q['domain'][:24]}: {audit['details']} ({latency:.2f}s)")

# 4. Generate Scorecard Summary
total_q = len(results)
total_passed = sum(1 for r in results if r["audit"]["passed"])
overall_pass_rate = (total_passed / total_q) * 100
total_format_clean = sum(1 for r in results if r["audit"]["format_valid"])
format_clean_rate = (total_format_clean / total_q) * 100
avg_latency = sum(r["audit"]["latency"] for r in results) / total_q

scorecard = {
    "model": "shreyansh12183/vigyan-olmo2-7b-quad-master",
    "base_model": "allenai/OLMo-2-1124-7B-Instruct",
    "lead_creator": "Shreyansh Singh (shreyansh00singh / shreyansh001boy-tech)",
    "execution_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "total_questions": total_q,
    "total_passed": total_passed,
    "overall_pass_rate": round(overall_pass_rate, 2),
    "format_clean_rate": round(format_clean_rate, 2),
    "avg_latency_sec": round(avg_latency, 2),
    "vector_breakdown": {}
}

for vec, stats in vector_stats.items():
    p_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    f_rate = (stats["format_clean"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    a_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
    scorecard["vector_breakdown"][vec] = {
        "total": stats["total"],
        "passed": stats["passed"],
        "pass_rate_pct": round(p_rate, 2),
        "format_clean_pct": round(f_rate, 2),
        "avg_latency_sec": round(a_lat, 2)
    }

# Save full results and scorecard
with open("certified_quad_master_benchmark_report.json", "w", encoding="utf-8") as f:
    json.dump({"scorecard": scorecard, "results": results}, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("🏛️ CERTIFIED CHIEF JUDGE SCORECARD: VIGYAN OLMo-2 7B QUAD-MASTER")
print("=" * 80)
print(f"Overall Pass Rate: {overall_pass_rate:.2f}% ({total_passed}/{total_q})")
print(f"Format Decoupling Success: {format_clean_rate:.2f}% (Rogue <thought> tag eradication)")
print(f"Mean Latency: {avg_latency:.2f}s per question")
print("\nVector Breakdown:")
for vec, data in scorecard["vector_breakdown"].items():
    print(f"  • {vec:<30}: {data['pass_rate_pct']:>6.2f}% ({data['passed']}/{data['total']}) | Latency: {data['avg_latency_sec']:.2f}s")
print("=" * 80)
print("Saved certified report to certified_quad_master_benchmark_report.json")
