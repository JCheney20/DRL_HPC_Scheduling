# Honours Submission 1 - 4 Minute Presentation

**Date:** April 2026  
**Duration:** 4:00 minutes  
**Audience:** Honours Committee + UWC Faculty  
**Format:** Recorded video presentation

---

## Slide 1: Problem Context & Literature Gap

**Visual Elements:**
- TOP500 accelerator growth chart (2006-2025)
- Literature survey statistics box
- Research implication callout

**Content:**

### HPC Scheduling Challenge
- Heterogeneous systems increasingly prevalent
- Traditional heuristics inflexible
- Meta-heuristics don't generalise
- DRL offers adaptive policy learning

### Critical Literature Gap
**Systematic Survey (59 papers, 2024-2025)**
- **27 papers (46%)** use PPO-based approaches
- **Only 12 papers (20%)** empirically justify this choice
- Most cite "stability" without comparative evidence

**Research Implication:**
Methodological convergence without empirical validation suggests **algorithmic path dependence**

**Speaker Notes:**
> "Modern HPC systems are increasingly heterogeneous, as shown by TOP500 data from 2006 to 2025. I conducted a systematic survey of 59 papers and found that 27 papers—46 percent—employed PPO or its variants. However, [PAUSE] only 12 of those 27 papers provided comparative empirical validation. This reveals algorithmic path dependence—researchers may be choosing PPO because prior work used PPO, not because it's demonstrably superior."

---

## Slide 2: Research Questions & Methodology

**Visual Elements:**
- Primary research question (highlighted box)
- Algorithm selection table
- Statistical framework flowchart

**Content:**

### Primary Research Question
> Is the field's preference for masked PPO variants empirically justified compared to other DRL algorithm families?

### Algorithm Selection

| Algorithm | Family | Masking |
|-----------|--------|---------|
| MaskablePPO | Policy Gradient | ✅ |
| MaskableDQN | Value-Based | ✅ |
| Vanilla PPO | Policy Gradient | ❌ |
| A2C | Actor-Critic | ❌ |

**Training Protocol:**
4 algorithms × 5 seeds × 2 real Slurm traces (84k + 28k jobs)

### Statistical Framework

1. **Shapiro-Wilk** → Normality check
2. **Friedman Test** → Omnibus significance
3. **Nemenyi Post-hoc** → Pairwise comparisons
4. **CD Diagrams + ε²** → Visualization + effect size

*Reference: Demšar 2006 methodology*

**Speaker Notes:**
> "My primary question: Is the field's preference for masked PPO empirically justified? I'm comparing four algorithms spanning three major DRL families. MaskablePPO and MaskableDQN both use action masking for fair comparison. Vanilla PPO is unmasked to isolate the masking benefit. Each algorithm will be trained with five independent seeds across two real Slurm traces. The statistical approach uses Friedman tests—appropriate because the same workloads run through all algorithms, creating related samples."

---

## Slide 3: Current Progress

**Visual Elements:**
- Completed infrastructure checklist
- Baseline results table

**Content:**

### Completed Infrastructure
✅ **Gymnasium simulation environment** (HPCsim with action masking)  
✅ **Classical baseline evaluation** (42 combinations across 2 partitions)  
✅ **Statistical analysis framework** (Friedman + Nemenyi pipeline)  
✅ **Paper writing** (Introduction + Literature Review complete)  
✅ **Visualization pipeline** (Multi-run analysis tools ready)

### Baseline Results (Reference Point)

**Physical Cluster (84k CPU jobs):**

| Algorithm | Avg Wait | Slowdown |
|-----------|----------|----------|
| LCFS + best_fit | 1,851s | 12.71 |
| UNICEP + best_fit | 1,955s | 12.78 |
| SJF + best_fit | 1,988s | 13.32 |
| FCFS + best_fit | 2,098s | 15.21 |

**Performance Target:**  
DRL must beat 1,851s avg wait to justify complexity

**Speaker Notes:**
> "Progress to date: The foundational infrastructure is complete. Classical baselines are fully evaluated—42 combinations across selectors and allocators. For reference, the best traditional approach—LCFS with best-fit—achieves 1,851 seconds average waiting time. Standard FCFS gets 2,098 seconds. These establish the performance bar DRL must exceed. On the writing side, Introduction and Literature Review sections are complete."

---

## Slide 4: Research Plan & Timeline

**Visual Elements:**
- Timeline Gantt chart (Feb-July 2026, teal bars)
- Next steps breakdown
- Computational resources box

**Content:**

### Timeline Overview
**Feb 2026 - July 2026** (4 major milestones)
- Literature Review ✅ Complete
- Infrastructure & Training 🔄 Current phase
- Results & Statistical Analysis ⏳ Planned
- Discussion & Final Writing ⏳ Planned

### Next Steps (6 Weeks - Submission 2)
1. **Weeks 1-2:** Setting up DRL training infrastructure
2. **Weeks 3-5:** DRL training & statistical analysis setup
3. **Week 6:** Writing Submission 2 & update presentation/website

### Computational Resources
- **Training Budget:** ~400 GPU-hours estimated
- **Wall-clock time:** ~9 days with parallelization
- **GPU allocation:** 2-week period (to be confirmed)

**Speaker Notes:**
> "The full timeline spans February through July with four major milestones. Literature Review is complete, we're currently in Infrastructure and Training phase. For the next six weeks: Weeks one and two focus on setting up DRL training infrastructure. Weeks three through five are for training runs and statistical analysis setup. Week six is dedicated to Submission 2. Computational requirements: approximately 400 GPU-hours, translating to about nine days wall-clock time with parallelization."

---

## Slide 5: Expected Contribution

**Visual Elements:**
- Primary contribution box (centered)
- Two outcome scenario boxes
- Broader impact statement

**Content:**

### Primary Contribution
> First rigorous statistical comparison of DRL algorithm families for HPC scheduling on real heterogeneous workloads

### Both Outcome Scenarios Are Valuable

**If PPO wins:**  
Empirical validation for field's convergence

**If alternative wins:**  
Challenge conventional assumptions

### Broader Impact
Foundation for distillation and deployment in resource-constrained environments

**Speaker Notes:**
> "Expected contribution: This is the first rigorous statistical comparison of DRL algorithm families for HPC scheduling on real heterogeneous workloads. Both outcome scenarios are valuable. If PPO wins, we provide empirical validation. If an alternative proves superior, we challenge conventional assumptions. Broader impact: This establishes foundations for distillation and deployment in resource-constrained environments—addressing practical constraints in African universities and similar institutions. Thank you."

---

## Timing Breakdown

| Slide | Duration | Cumulative |
|-------|----------|------------|
| 1 | 60s | 1:00 |
| 2 | 75s | 2:15 |
| 3 | 60s | 3:15 |
| 4 | 45s | 4:00 |
| 5 | 30s | 4:30 |
| **Buffer** | **-30s** | **4:00** |

---

## Key Emphasis Points

- "**27 papers**" and "**46 percent**"
- "**only 12 papers**" [PAUSE]
- "**algorithmic path dependence**"
- "**empirically justified**"
- "**related samples**"
- "**1,851 seconds**"
- "**first rigorous statistical comparison**"

---

## Figures

- `top500_accelerator_growth.pdf` - Heterogeneous systems growth (2006-2025)
- `timeline_condensed.pdf` - Research Gantt chart (teal bars, Feb-July 2026)

---

**Presentation Style:**
- Natural pacing with strategic pauses
- Gesture to visuals when referencing
- Eye contact with camera
- Confident delivery (research is solid)
- End strongly (not with "um, yeah, that's it")
