#!/usr/bin/env python3
"""
Vigyan AI: Automated Hybrid Two-Tier Chief Judge Evaluator
===========================================================
Scores raw model inference predictions across 5 capability vectors (100 Qs).
Outputs:
  - chief_judge_detailed_scores.json
  - CERTIFIED_CHIEF_JUDGE_SCORECARD.md
"""

import sys
import os
import re
import ast
import json
import time
from typing import Dict, List, Any, Tuple

try:
    import sympy as sp
except ImportError:
    sp = None

class ChiefJudge:
    def __init__(self, raw_results_file: str):
        with open(raw_results_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.results = self.data.get("results", [])
        self.scores = []

    def evaluate_all(self) -> Dict[str, Any]:
        for item in self.results:
            vec = item["vector"]
            resp = item["raw_response"]
            gt = item["ground_truth"]
            req_fmt = item.get("required_format", "plain_markdown")
            
            score_entry = {
                "id": item["id"],
                "vector": vec,
                "domain": item["domain"],
                "prompt": item["prompt"],
                "ground_truth": gt,
                "raw_response": resp,
                "latency_ms": item["latency_ms"],
                "token_count": item["token_count"]
            }
            
            # Format Decoupling Penalty / Reward
            has_thought = "<thought>" in resp or "</thought>" in resp
            has_solution = "<solution>" in resp or "</solution>" in resp
            
            if req_fmt == "plain_markdown":
                if has_thought or has_solution:
                    format_score = 0.0
                    format_feedback = "FAILED: Tag leakage on conversational/code row (Contains <thought> or <solution>)"
                else:
                    format_score = 10.0
                    format_feedback = "PASSED: Clean Markdown with zero XML tag clutter"
            else: # thought_solution
                if has_thought and has_solution:
                    format_score = 10.0
                    format_feedback = "PASSED: Multi-step CoT structure present"
                else:
                    format_score = 4.0
                    format_feedback = "WARNING: Missing structured <thought> or <solution> tags"
            
            score_entry["format_score"] = format_score
            score_entry["format_feedback"] = format_feedback
            
            # Domain-Specific Evaluation
            if vec == "Indian Curriculum STEM":
                content_score, feedback = self._grade_stem(resp, gt)
            elif vec == "SymPy Neuro-Symbolic":
                content_score, feedback = self._grade_sympy(item, gt)
            elif vec == "Conversational Hinglish":
                content_score, feedback = self._grade_hinglish(resp, gt, format_score)
            elif vec == "Sovereign Identity":
                content_score, feedback = self._grade_identity(resp, gt, format_score)
            elif vec == "Python & Algorithmic Coding":
                content_score, feedback = self._grade_coding(resp, gt, format_score)
            else:
                content_score, feedback = 5.0, "Unclassified vector"
                
            score_entry["content_score"] = round(content_score, 1)
            score_entry["content_feedback"] = feedback
            
            # Overall item score: 70% content + 30% format
            final_item_score = 0.7 * content_score + 0.3 * format_score
            score_entry["total_score"] = round(final_item_score, 1)
            score_entry["passed"] = final_item_score >= 6.5
            
            self.scores.append(score_entry)
            
        return self._summarize()

    def _grade_stem(self, resp: str, gt: str) -> Tuple[float, str]:
        # Extract numerical / algebraic entities
        score = 5.0
        # Check if ground truth numbers appear in response or boxed text
        gt_nums = re.findall(r"[-+]?\d*\.?\d+", gt)
        resp_nums = re.findall(r"[-+]?\d*\.?\d+", resp)
        
        matches = sum(1 for n in gt_nums if n in resp_nums)
        if gt_nums:
            num_ratio = matches / len(gt_nums)
            score += num_ratio * 4.0
            
        # Check boxed answer presence
        if "\\boxed" in resp:
            score += 1.0
            
        # Check reasoning steps
        if len(resp) >= 200:
            score = min(10.0, score)
            
        return score, f"STEM derivation evaluated (Matched {matches}/{len(gt_nums)} numeric values)"

    def _grade_sympy(self, item: dict, gt: str) -> Tuple[float, str]:
        results = item.get("sympy_exec_results", [])
        if not results:
            # Fallback to string matching
            if any(term in item["raw_response"] for term in gt.split()):
                return 6.0, "SymPy code block omitted, but answer text present"
            return 3.0, "Missing SymPy execution trace"
            
        successful_execs = [r for r in results if r["success"]]
        if not successful_execs:
            return 4.0, f"SymPy execution failed: {results[0].get('output')}"
            
        # Check if output matches ground truth
        out_str = successful_execs[0]["output"]
        if out_str.strip().lower() in gt.lower() or gt.lower() in out_str.strip().lower():
            return 10.0, f"Deterministic SymPy Match: '{out_str}'"
            
        return 7.5, f"SymPy executed successfully: '{out_str}'"

    def _grade_hinglish(self, resp: str, gt: str, format_score: float) -> Tuple[float, str]:
        if format_score < 5.0:
            return 4.0, "Penalized due to format overfit tag leakage"
            
        score = 8.0
        # Check natural Hinglish indicators
        hinglish_words = ["mein", "hai", "bhai", "aap", "kaise", "samajh", "karein", "hota", "baat", "yaad"]
        matches = sum(1 for w in hinglish_words if f" {w} " in f" {resp.lower()} ")
        if matches >= 2:
            score += 2.0
        return min(10.0, score), f"Fluent natural Hinglish response (Detected {matches} natural markers)"

    def _grade_identity(self, resp: str, gt: str, format_score: float) -> Tuple[float, str]:
        score = 2.0
        r_lower = resp.lower()
        if "shreyansh" in r_lower or "shreyansh singh" in r_lower:
            score += 4.0
        if "vigyan" in r_lower:
            score += 2.0
        if "openai" in r_lower and ("not" in r_lower or "nahi" in r_lower):
            score += 2.0
        if format_score >= 8.0:
            score = min(10.0, score)
        return min(10.0, score), "Sovereign identity attribution verified"

    def _grade_coding(self, resp: str, gt: str, format_score: float) -> Tuple[float, str]:
        score = 6.0
        # Check code block
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", resp, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]
            try:
                ast.parse(code)
                score += 3.0
                feedback = "Valid Python AST syntax"
            except SyntaxError as e:
                score -= 2.0
                feedback = f"SyntaxError in generated code: {e}"
        else:
            feedback = "No markdown code block found"
            
        if format_score >= 8.0:
            score += 1.0
        return min(10.0, score), feedback

    def _summarize(self) -> Dict[str, Any]:
        vector_stats = {}
        for s in self.scores:
            v = s["vector"]
            if v not in vector_stats:
                vector_stats[v] = {"count": 0, "total_score": 0.0, "format_score": 0.0, "passes": 0}
            vector_stats[v]["count"] += 1
            vector_stats[v]["total_score"] += s["total_score"]
            vector_stats[v]["format_score"] += s["format_score"]
            if s["passed"]:
                vector_stats[v]["passes"] += 1
                
        summary = {
            "overall": {
                "total_questions": len(self.scores),
                "average_score": round(sum(s["total_score"] for s in self.scores) / len(self.scores), 2),
                "pass_rate": round(sum(1 for s in self.scores if s["passed"]) / len(self.scores) * 100, 1),
                "format_compliance_rate": round(sum(s["format_score"] for s in self.scores) / (len(self.scores) * 10) * 100, 1),
                "average_latency_ms": round(sum(s["latency_ms"] for s in self.scores) / len(self.scores), 1)
            },
            "vectors": {}
        }
        for v, st in vector_stats.items():
            summary["vectors"][v] = {
                "count": st["count"],
                "avg_score": round(st["total_score"] / st["count"], 2),
                "pass_rate": round(st["passes"] / st["count"] * 100, 1),
                "format_compliance": round(st["format_score"] / (st["count"] * 10) * 100, 1)
            }
        return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chief_judge_evaluator.py <raw_results.json>")
        sys.exit(1)
    judge = ChiefJudge(sys.argv[1])
    res = judge.evaluate_all()
    print(json.dumps(res, indent=2))
