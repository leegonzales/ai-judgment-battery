# AI Judgment Battery - Experimental Plan

## Research Question

**Do frontier AI models differ meaningfully in ethical reasoning quality?**

Specifically: When controlling for confounding variables (length, brand bias, position), which model provides the highest quality ethical reasoning for real-world dilemmas?

---

## Models Under Test

| Provider | Model | ID |
|----------|-------|-----|
| Anthropic | Claude Opus 4.5 | `claude-opus-4-5-20251101` |
| OpenAI | GPT-5.1 | `gpt-5.1` |
| Google | Gemini 3 Pro | `gemini-3-pro-preview` |

---

## Confounding Variables Identified

### 1. Response Length Bias
**Problem:** GPT-5.1 produces ~12,600 chars avg, Gemini ~5,400 chars, Claude ~2,200 chars. Longer responses may appear more thorough.

**Evidence:** When Sonnet judged, GPT-5.1 won 58% (always longest). When Opus judged, Gemini won 78% (middle length).

**Solution:** Add word limit to system prompt: "Keep your response focused and concise: aim for 800-1200 words"

### 2. Brand/Name Bias
**Problem:** Judge sees "claude-opus", "gpt-5.1" etc. May have preconceptions.

**Solution:** Blind model names to "Model A", "Model B", "Model C" before judging.

### 3. Position Bias
**Problem:** Fixed presentation order may favor first or last response.

**Solution:** Randomize order for each comparison.

### 4. Judge Identity Bias
**Problem:** Different judges give wildly different results:
- Sonnet judge: GPT-5.1 wins 58%
- Opus judge: Gemini wins 78%

**Solution:** Use multiple judges and report variance. Consider using non-Anthropic judge (GPT-5.1) for balance.

### 5. Judge Prompt Framing
**Problem:** Original prompt didn't explicitly penalize length-padding or reward concision.

**Solution:** Updated prompt with explicit anti-length-bias instructions:
- "Judge QUALITY of reasoning, not QUANTITY"
- "Response length is NOT a factor"
- "concise excellence beats verbose mediocrity"

---

## Experimental Design

### Phase 1: Length-Controlled Data Collection
Re-run all three batteries with updated system prompt containing word limit.

```bash
python harness/run_battery.py --model claude-opus --workers 5
python harness/run_battery.py --model gpt-5.1 --workers 5
python harness/run_battery.py --model gemini-3-pro --workers 5
```

**Expected output:** 50 responses per model, each 800-1200 words (~4000-6000 chars)

### Phase 2: Blinded Comparison with Multiple Judges

Run comparison with each judge:

```bash
# Sonnet as judge
python harness/compare.py --models claude-opus gpt-5.1 gemini-3-pro --judge claude-sonnet --workers 5

# Opus as judge
python harness/compare.py --models claude-opus gpt-5.1 gemini-3-pro --judge claude-opus --workers 5

# GPT-5.1 as judge (cross-provider)
python harness/compare.py --models claude-opus gpt-5.1 gemini-3-pro --judge gpt-5.1 --workers 5
```

### Phase 3: Analysis

1. **Decode blind labels** using `blind_mapping` in results
2. **Calculate win rates** per model per judge
3. **Check inter-judge reliability** - do different judges agree?
4. **Analyze by category** - do models excel at different dilemma types?
5. **Statistical significance** - is the difference real or noise?

---

## Success Criteria

1. **Response lengths normalized** - all models within 4000-6000 chars
2. **Judge agreement** - multiple judges rank similarly (>60% agreement on winner)
3. **Clear differentiation** - one model consistently outperforms OR models have different strengths by category

---

## Current Status

### Completed
- [x] Initial battery runs (all 3 models, 50 dilemmas each)
- [x] Identified length bias problem
- [x] Fixed GPT-5.1 empty response issue (reasoning tokens)
- [x] Fixed Gemini truncation issue (token limit)
- [x] Upgraded to Gemini 3 Pro
- [x] Built comparison harness with judge prompt
- [x] Discovered judge identity affects results dramatically
- [x] Implemented blinded evaluation (Model A/B/C)
- [x] Added anti-length-bias instructions to judge prompt
- [x] Added randomized presentation order

### Next Steps
- [ ] Re-run all 3 batteries with length-controlled prompt
- [ ] Run blinded comparison with Sonnet judge
- [ ] Run blinded comparison with Opus judge
- [ ] Run blinded comparison with GPT-5.1 judge
- [ ] Analyze results and report findings

---

## Files

| File | Purpose |
|------|---------|
| `harness/run_battery.py` | Run dilemmas through models |
| `harness/compare.py` | Cross-model comparison with blinded judge |
| `harness/analyze.py` | Score individual responses |
| `results/` | Raw model responses |
| `comparisons/` | Judge comparisons |
| `dilemmas/all_dilemmas.json` | 50 ethical dilemmas |

---

## Commands Reference

```bash
# List available models
python harness/run_battery.py --list-models

# Run battery (latest models)
python harness/run_battery.py --model claude-opus
python harness/run_battery.py --model gpt-5.1
python harness/run_battery.py --model gemini-3-pro

# Run comparison (blinded, anti-length-bias)
python harness/compare.py --models claude-opus gpt-5.1 gemini-3-pro --judge claude-sonnet

# Analyze results
python harness/analyze.py --latest
```
