# Comparative Report: Speech Detection Models with OPRO

## Executive Summary

This report compares three model configurations for speech/non-speech detection:
1. **Qwen2.5-Audio-7B (Base)** - Base model without fine-tuning
2. **Qwen2.5-Audio-7B (LoRA)** - Fine-tuned with LoRA adapters
3. **Qwen3-Omni-30B** - Larger multimodal model (no fine-tuning)

Each configuration was evaluated with:
- Hand-crafted baseline prompt
- OPRO-optimized prompt

---

## Main Results Table

| Model | Prompt Type | BA_clip | Speech Acc | Nonspeech Acc | Δ from Baseline |
|-------|-------------|---------|------------|---------------|-----------------|
| **Qwen2.5-Audio-7B (Base)** | Hand-crafted | 64.06% | 32.25% | 95.88% | - |
| **Qwen2.5-Audio-7B (Base)** | OPRO | 88.12% | 91.64% | 84.60% | **+24.06%** |
| **Qwen2.5-Audio-7B (LoRA)** | Hand-crafted | 93.02% | 98.35% | 87.69% | - |
| **Qwen2.5-Audio-7B (LoRA)** | OPRO | 94.90% | 98.23% | 91.57% | **+1.88%** |
| **Qwen3-Omni-30B** | Hand-crafted | 91.09% | 87.29% | 94.88% | - |
| **Qwen3-Omni-30B** | OPRO | 87.51% | 97.91% | 77.10% | **-3.58%** |

---

## Detailed Analysis by Model

### 1. Qwen2.5-Audio-7B (Base Model)

**Configuration**: Base model, no fine-tuning, 7B parameters

| Metric | Hand-crafted | OPRO | Delta |
|--------|--------------|------|-------|
| BA_clip | 64.06% | 88.12% | +24.06% |
| BA_conditions | - | - | - |
| Speech Recall | 32.25% | 91.64% | +59.39% |
| Nonspeech Recall | 95.88% | 84.60% | -11.28% |

**Observation**: OPRO provides massive improvement (+24%) by fixing the severe speech detection bias. Base model with hand-crafted prompt fails to detect most speech samples.

---

### 2. Qwen2.5-Audio-7B (LoRA Fine-tuned)

**Configuration**: Fine-tuned with LoRA adapters, 7B parameters

| Metric | Hand-crafted | OPRO | Delta |
|--------|--------------|------|-------|
| BA_clip | 93.02% | 94.90% | +1.88% |
| BA_conditions | - | - | - |
| Speech Recall | 98.35% | 98.23% | -0.12% |
| Nonspeech Recall | 87.69% | 91.57% | +3.88% |

**Observation**: LoRA fine-tuning already achieves strong performance. OPRO provides modest but consistent improvement (+1.88%), mainly by improving nonspeech detection.

---

### 3. Qwen3-Omni-30B (No Fine-tuning)

**Configuration**: Base model, no fine-tuning, 30B parameters (Mixture of Experts)

| Metric | Hand-crafted | OPRO | Delta |
|--------|--------------|------|-------|
| BA_clip | 91.09% | 87.51% | **-3.58%** |
| BA_conditions | 93.01% | 89.65% | -3.36% |
| Speech Recall | 87.29% | 97.91% | +10.62% |
| Nonspeech Recall | 94.88% | 77.10% | -17.78% |

**Dimension Breakdown (Baseline)**:
| Dimension | BA |
|-----------|-----|
| Duration | 79.68% |
| SNR | 98.49% |
| Reverb | 97.19% |
| Filter | 96.70% |

**Observation**: Qwen3-Omni achieves excellent baseline performance (91.09% BA) without any fine-tuning. However, OPRO optimization **decreased** performance (-3.58%) by over-optimizing for speech detection at the cost of nonspeech accuracy.

---

## Cross-Model Comparison

### Best Configurations Ranked by BA_clip

| Rank | Model + Prompt | BA_clip | 95% CI |
|------|----------------|---------|--------|
| 1 | Qwen2.5-LoRA + OPRO | **94.90%** | [94.17%, 95.58%] |
| 2 | Qwen2.5-LoRA + Hand | 93.02% | [92.08%, 93.87%] |
| 3 | **Qwen3-Omni + Hand** | **91.09%** | - |
| 4 | Qwen2.5-Base + OPRO | 88.12% | [86.84%, 89.33%] |
| 5 | Qwen3-Omni + OPRO | 87.51% | - |
| 6 | Qwen2.5-Base + Hand | 64.06% | [62.70%, 65.41%] |

### Key Insights

1. **Best Overall**: Qwen2.5-LoRA + OPRO (94.90%)
   - Fine-tuning + prompt optimization yields best results

2. **Best Without Fine-tuning**: Qwen3-Omni + Hand-crafted (91.09%)
   - Larger model compensates for lack of task-specific training
   - Achieves 91%+ without any adaptation

3. **OPRO Effectiveness Varies**:
   - Qwen2.5-Base: +24.06% (massive improvement)
   - Qwen2.5-LoRA: +1.88% (modest improvement)
   - Qwen3-Omni: -3.58% (degradation)

4. **Model Size vs Fine-tuning**:
   - Qwen3-Omni (30B, no fine-tuning): 91.09%
   - Qwen2.5-LoRA (7B, fine-tuned): 93.02%
   - Fine-tuning on smaller model outperforms larger base model

---

## OPRO Prompts

### Qwen2.5-Base OPRO Prompt
```
"Listen to the audio. Is there human speech? Reply SPEECH or NONSPEECH."
```
*Improvement: +24.06%*

### Qwen2.5-LoRA OPRO Prompt
```
"Classify: SPEECH if human voice detected, NONSPEECH otherwise."
```
*Improvement: +1.88%*

### Qwen3-Omni OPRO Prompt
```
"Determine: SPEECH if human talking, else NONSPEECH."
```
*Degradation: -3.58%*

---

## Conclusions

1. **Fine-tuning remains superior**: Qwen2.5-LoRA + OPRO achieves the best results (94.90%), showing that task-specific adaptation is still valuable.

2. **Qwen3-Omni is impressive zero-shot**: Without any fine-tuning, Qwen3-Omni achieves 91.09% BA_clip, demonstrating strong generalization from its larger scale.

3. **OPRO is not universally beneficial**: While OPRO dramatically helps underperforming models (Qwen2.5-Base: +24%), it can hurt already well-calibrated models (Qwen3-Omni: -3.58%).

4. **Recommendation**:
   - For best accuracy: Use Qwen2.5-LoRA + OPRO
   - For zero-shot deployment: Use Qwen3-Omni with hand-crafted prompt
   - For resource-constrained settings: Use Qwen2.5-Base + OPRO

---

## Technical Details

### Evaluation Dataset
- **Total samples**: 21,340
- **Speech samples**: 10,670
- **Non-speech samples**: 10,670
- **Conditions**: 22 unique degradation conditions
- **Dimensions**: Duration, SNR, Reverb, Filter

### OPRO Configuration
- **Iterations**: 15
- **Candidates per iteration**: 8
- **Samples per iteration**: 20
- **Optimizer model**: Qwen2.5-7B-Instruct
- **Temperature**: 0.7

### Compute Resources
- **GPU**: NVIDIA A100-SXM4-80GB
- **Qwen3-Omni evaluation time**: ~6.5 hours (including OPRO)

---

*Report generated: 2026-01-17*
*Pipeline: opro2_clean/paper_artifacts_qwen3*
