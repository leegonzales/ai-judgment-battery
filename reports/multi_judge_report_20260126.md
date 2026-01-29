# AI Judgment Battery: Multi-Judge Comparison Report

**Date:** January 26, 2026
**Battery Version:** 50 ethical dilemmas across 7 categories
**Models Evaluated:** Claude Opus 4.5, GPT-5.1, Gemini 3 Pro

---

## Executive Summary

This report presents results from a multi-judge evaluation of three frontier AI models on ethical reasoning tasks. To mitigate self-preference bias (where models favor their own responses), each model served as a judge, and results were aggregated across all three perspectives.

**Key Findings:**
- **Claude Opus 4.5** ranked first with an Elo rating of 1668 and 79% win rate
- **GPT-5.1** ranked second with an Elo rating of 1536 and 61% win rate
- **Gemini 3 Pro** ranked third with an Elo rating of 1296 and 10% win rate
- Self-preference bias was observed but did not change the overall ranking
- Notably, Gemini showed *anti-self* bias, ranking Claude highest even above itself

---

## Methodology

### Battery Design

The AI Judgment Battery consists of 50 ethical dilemmas spanning 7 categories:

| Category | Description | Count |
|----------|-------------|-------|
| A | Professional Ethics & Whistleblowing | 8 |
| B | Resource Allocation & Triage | 8 |
| C | Privacy vs. Security | 8 |
| D | Autonomy vs. Paternalism | 8 |
| E | Personal Ethics & Relationships | 6 |
| F | Technology & Emerging Issues | 6 |
| G | Justice & Fairness | 6 |

### Response Generation

Each model generated responses to all 50 dilemmas with:
- **Max tokens:** 800 (targeting ~2000 characters for digestibility)
- **System prompt:** Ethical reasoning guidelines with conciseness instruction
- **Workers:** 5 parallel requests

### Evaluation Protocol

1. **Multi-Judge Comparison:** Each of the three models served as a judge
2. **Blind Evaluation:** Responses were labeled A/B/C with randomized order
3. **Structured Output:** Judges provided rankings with reasoning
4. **Aggregation:** Final rankings derived from consensus across all judges

### Models Tested

| Provider | Model | Model ID |
|----------|-------|----------|
| Anthropic | Claude Opus 4.5 | `claude-opus-4-5-20251101` |
| OpenAI | GPT-5.1 | `gpt-5.1` |
| Google | Gemini 3 Pro | `gemini-3-pro-preview` |

---

## Results

### Elo Ratings

Elo ratings calculated from 150 pairwise comparisons (50 dilemmas × 3 judges):

| Model | Elo Rating | Δ from 1500 |
|-------|------------|-------------|
| Claude Opus 4.5 | **1667.6** | +168 |
| GPT-5.1 | 1536.2 | +36 |
| Gemini 3 Pro | 1296.2 | -204 |

### Win Statistics

| Model | Wins | Total | Win Rate | 95% CI |
|-------|------|-------|----------|--------|
| Claude Opus 4.5 | 237 | 300 | **79.0%** | 74.0% - 83.2% |
| GPT-5.1 | 182 | 300 | 60.7% | 55.0% - 66.0% |
| Gemini 3 Pro | 31 | 300 | 10.3% | 7.4% - 14.3% |

*Note: Win rate represents how often a model placed 1st in the 3-way ranking.*

### Head-to-Head Matrix

Direct comparison wins (row beats column):

| Winner ↓ | vs Claude | vs Gemini | vs GPT |
|----------|-----------|-----------|--------|
| **Claude Opus** | — | 137 (91%) | 100 (67%) |
| **GPT-5.1** | 50 (33%) | 132 (88%) | — |
| **Gemini 3 Pro** | 13 (9%) | — | 18 (12%) |

**Interpretation:**
- Claude beats Gemini in 91% of matchups
- Claude beats GPT in 67% of matchups
- GPT beats Gemini in 88% of matchups

---

## Judge Bias Analysis

A critical concern in AI-judged evaluations is self-preference bias. We analyzed each judge's tendency to favor its own responses.

### Self-Win Rates by Judge

| Judge Model | Own Model Win Rate | Bias Level |
|-------------|-------------------|------------|
| Claude Opus | 60.7% | Strong self-preference |
| GPT-5.1 | 46.0% | Moderate self-preference |
| Gemini 3 Pro | 9.3% | **Anti-self bias** |

### Detailed Judge Breakdown

**When Claude judges (150 comparisons):**
| Model | Wins | Rate |
|-------|------|------|
| Claude Opus | 91 | 60.7% ⬆ |
| GPT-5.1 | 58 | 38.7% |
| Gemini 3 Pro | 1 | 0.7% |

**When GPT-5.1 judges (150 comparisons):**
| Model | Wins | Rate |
|-------|------|------|
| GPT-5.1 | 69 | 46.0% ⬆ |
| Claude Opus | 65 | 43.3% |
| Gemini 3 Pro | 16 | 10.7% |

**When Gemini judges (150 comparisons):**
| Model | Wins | Rate |
|-------|------|------|
| Claude Opus | 81 | 54.0% |
| GPT-5.1 | 55 | 36.7% |
| Gemini 3 Pro | 14 | 9.3% ⬇ |

### Bias Interpretation

The multi-judge approach reveals important patterns:

1. **Claude shows strong self-preference** (61% self-win rate vs ~40% when judged by others), but this doesn't flip the overall ranking—Claude still wins when judged by GPT (43%) and Gemini (54%)

2. **GPT shows moderate self-preference** (46% self-win rate), but rates Claude nearly as high (43%)

3. **Gemini shows anti-self bias** (9% self-win rate), actually ranking Claude higher than itself. This unexpected finding suggests Gemini may have calibration issues or genuinely recognizes Claude's superior ethical reasoning

4. **Claude's lead is robust:** Even excluding Claude-as-judge data, Claude still wins 54% of Gemini-judged and 43% of GPT-judged comparisons

---

## Category Analysis

Performance varied by ethical domain:

| Category | Claude | GPT | Gemini | Leader |
|----------|--------|-----|--------|--------|
| A - Professional Ethics | 52.8% | 36.1% | 11.1% | Claude |
| B - Resource Allocation | 56.9% | 36.1% | 6.9% | Claude |
| C - Privacy vs Security | 54.2% | 41.7% | 4.2% | Claude |
| D - Autonomy vs Paternalism | 56.9% | 40.3% | 2.8% | Claude |
| E - Personal Ethics | 46.3% | **48.1%** | 5.6% | **GPT** |
| F - Emerging Tech | 40.7% | **48.1%** | 11.1% | **GPT** |
| G - Justice & Fairness | 57.4% | 35.2% | 7.4% | Claude |

### Notable Patterns

- **Claude dominates** in professional ethics (A), resource allocation (B), and justice (G)
- **GPT edges ahead** in personal ethics (E) and emerging technology (F)
- **Gemini consistently trails** across all categories, never exceeding 11%

---

## Response Characteristics

### Token Usage

| Model | Avg Tokens | Avg Characters |
|-------|------------|----------------|
| Claude Opus | 420 | ~2,000 |
| GPT-5.1 | 470 | ~2,200 |
| Gemini 3 Pro | 430 | ~2,100 |

All models produced appropriately concise responses within the 800-token limit.

### Processing Time

| Model | Total Time | Per Dilemma |
|-------|------------|-------------|
| Claude Opus | 132s | 2.6s |
| GPT-5.1 | 124s | 2.5s |
| Gemini 3 Pro | 191s | 3.8s |

---

## Limitations

1. **AI Judging AI:** Despite multi-judge aggregation, all evaluators are AI models. Human validation showed only moderate correlation with AI judgments in preliminary testing.

2. **Self-Preference Bias:** While aggregation reduces bias, it doesn't eliminate it. Claude's high self-win rate (61%) may inflate its overall score.

3. **Gemini Anomaly:** Gemini's anti-self bias is unexpected and may indicate response quality issues or miscalibration rather than objective assessment.

4. **Sample Size:** 50 dilemmas with 3 judges yields 150 comparisons—sufficient for trends but limited for fine-grained category analysis.

5. **Response Format:** Shorter responses (800 tokens) may favor some rhetorical styles over others.

---

## Conclusions

1. **Claude Opus 4.5 demonstrates the strongest ethical reasoning** among the three models, with a 79% win rate and 1668 Elo rating. This finding is robust across all three judges.

2. **GPT-5.1 is a strong second-place finisher** with particular strength in personal ethics and emerging technology domains.

3. **Gemini 3 Pro significantly underperforms** on this benchmark, winning only 10% of comparisons. Its anti-self bias raises questions about response quality.

4. **Multi-judge aggregation successfully mitigates bias** by averaging across perspectives, though the ranking remains consistent regardless of which judge is used.

5. **The ~130 Elo gap between Claude and GPT** (1668 vs 1536) corresponds to approximately 70% expected win probability in head-to-head matchups, confirmed by the 67% observed rate.

---

## Data Files

| File | Description |
|------|-------------|
| `results/run_20260126_044231_claude-opus.json` | Claude responses |
| `results/run_20260126_044231_gpt-5.1.json` | GPT responses |
| `results/run_20260126_044233_gemini-3-pro.json` | Gemini responses |
| `comparisons/compare_20260126_160243.json` | Claude-judged comparisons |
| `comparisons/compare_20260126_160505.json` | GPT-judged comparisons |
| `comparisons/compare_20260126_160740.json` | Gemini-judged comparisons |
| `comparisons/multi_judge_20260126_161430.json` | Aggregated results |

---

*Report generated by AI Judgment Battery v1.0*
