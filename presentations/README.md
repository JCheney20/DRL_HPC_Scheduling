# Presentation Archive

This directory contains markdown versions of all presentations delivered during the Honours research project.

---

## 📁 Structure

```
presentations/
├── submission1/        # Honours Submission 1 (4 minutes)
├── submission2/        # Honours Submission 2 (progress update)
└── symposium/          # Honours Symposium (10 minutes)
```

---

## 📋 Presentations

### Submission 1 (4 Minutes)

**Date:** [To be filled]  
**Audience:** Honours Committee + UWC Faculty  
**Focus:** Problem statement, literature gap, proposed methodology, current progress

**Files:**
- `submission1/slides.md` - Full slide content (Markdown)
- `submission1/script.md` - Speaker notes and timing
- `submission1/slides.pdf` - Compiled presentation (Typst output)

**Key Points:**
- Literature survey: 27/59 papers (46%) use PPO, only 12 (20%) justify empirically
- Primary RQ: Is PPO preference empirically justified?
- 4 algorithms × 5 seeds × 2 traces = 40 training runs
- Statistical framework: Friedman → Nemenyi → CD diagrams

---

### Submission 2 (Progress Update)

**Date:** [To be filled - Week 6]  
**Audience:** Honours Committee  
**Focus:** Training results, preliminary statistical analysis

**Status:** Planned

**Expected Content:**
- Training infrastructure implementation
- First round of DRL results
- Initial Friedman test outcomes
- Challenges encountered and resolutions

---

### Symposium (10 Minutes)

**Date:** [To be filled]  
**Audience:** General academic audience (supervisors + peers)  
**Focus:** Complete research overview from problem to impact

**Files:**
- `symposium/slides.md` - Full slide content
- `symposium/script.md` - Detailed speaker notes
- `symposium/slides.pdf` - Compiled presentation

**Key Sections:**
1. Problem: HPC scheduling complexity
2. DRL formulation and literature trends
3. Research questions and objectives
4. Methodology (algorithms, evaluation, statistics)
5. Current progress and timeline
6. Expected contributions and broader impact

---

## 🎯 Converting to Markdown

All presentations are originally created in **Typst** (`.typ` files) using the **Metropolyst theme**.

### Conversion Process

1. **Extract slide content** from `.typ` source
2. **Format as Markdown** with headers, bullets, tables
3. **Include speaker notes** for reproducibility
4. **Add images** (export figures to PNG/PDF)

### Example Structure

```markdown
# Slide 1: Title

**On Screen:**
- Project title
- Student name
- Institution
- Date

**Speaker Notes:**
"Good day. I'm Justin Cheney, presenting my Honours research on..."

---

# Slide 2: Problem Context

**On Screen:**
- TOP500 growth figure
- Literature statistics

**Speaker Notes:**
"Modern HPC systems are increasingly heterogeneous..."
```

---

## 📊 Figures

Figures referenced in presentations are stored in:
- `/Presentation/figures/` (source)
- `presentations/<type>/figures/` (copies for archival)

**Key Figures:**
- `top500_accelerator_growth.pdf` - Heterogeneous systems growth
- `timeline_condensed.pdf` - Research timeline (teal Gantt chart)
- `cd_diagram_*.pdf` - Critical Difference diagrams (future)

---

## 🎤 Presentation Best Practices

### Timing Guidelines
- **4-minute presentation:** ~50s per slide (5 content slides)
- **10-minute presentation:** ~75s per slide (8 content slides)
- Build in **buffer time** for natural pacing

### Emphasis Points
- Pause after key statistics (e.g., "46%", "only 20%")
- Gesture to visuals when referencing them
- Make eye contact (camera for recorded, audience for live)
- Avoid filler words ("um", "like", "so")

### Common Pitfalls
- ❌ Rushing through methodology (your credibility anchor)
- ❌ Reading slides verbatim
- ❌ Going over time (academic audiences notice)
- ❌ Apologizing for scope ("only 4 algorithms")

---

## 📧 Contact

For questions about presentation content or to request slides:

**Justin M. Cheney**  
University of the Western Cape  
📧 [Email TBD]
