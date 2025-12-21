# Image Suggestions for Presentation Slides

## **SLIDE 1: Opening - The Biological Question**

**Image Options:**
1. **Simple schematic diagram** (create in PowerPoint/Illustrator)
   - Show: Sleep is multidimensional
   - Visual: Venn diagram or overlapping circles labeled: "Duration", "Timing", "Fragmentation", "Circadian"
   - Why: Immediately shows complexity without technical details

2. **OR: Photo of DAM system** (if available)
   - Why: Grounds the talk in real equipment they use

**Recommendation:** Use the schematic - it's conceptual and sets up the problem visually

---

## **SLIDE 2: The Challenge - Why We Need Computational Tools**

**Image Options:**
1. **Data volume visualization** (create simple graphic)
   - Show: 64 flies × 7 days × 1,440 minutes = 645,120 data points
   - Visual: Stack of data points or spreadsheet icon with large number
   - Why: Makes the scale tangible

2. **OR: Comparison diagram**
   - Left side: "Manual analysis" (person with calculator, takes days)
   - Right side: "Automated pipeline" (computer, takes minutes)
   - Why: Shows the time savings clearly

**Recommendation:** Use the comparison diagram - emphasizes the practical benefit

---

## **SLIDE 3: What We've Built - The Pipeline Overview**

**MUST HAVE IMAGE:** Simple flowchart diagram

**Create this:**
```
[Raw DAM Data] 
    ↓
[Data Processing: Align to ZT, Filter MT]
    ↓
[Sleep Detection: Identify 5-min bouts]
    ↓
[Feature Extraction: 21 metrics]
    ↓
[Biological Insights]
```

**Visual Style:**
- Use boxes with rounded corners
- Arrows between steps
- Color-code: Blue (data) → Green (processing) → Orange (features) → Purple (insights)
- Keep it simple - no code, just concepts

**Why:** Shows the pipeline at a glance without overwhelming detail

**ADDITIONAL IMAGE FOR POINTS 3 & 4 (Feature Extraction & Automated Analysis):**

**Option A: Feature Hierarchy Diagram** (Recommended)
```
Show a hierarchical structure:

[Activity Data / Sleep Bouts]
    ↓
[Feature Extraction]
    ├─→ [17 Sleep Metrics]
    │   • Total sleep, Light/Dark sleep
    │   • Bout count, Mean bout length
    │   • Sleep efficiency, Latency
    │   • Transition probabilities
    │
    └─→ [4 Circadian Parameters]
        • Mesor (baseline)
        • Amplitude (rhythm strength)
        • Phase (peak timing)
        • p-value (rhythmicity)

    ↓
[Automated Analysis]
    ├─→ [Per-Fly Summaries]
    │   • One row per fly
    │   • Averaged across days
    │
    └─→ [Group-Level Summaries]
        • Mean ± SEM by group
        • Genotype × Sex × Treatment
        • Ready for statistics
```

**Visual Style:**
- Tree/hierarchical layout
- Color-code: Sleep metrics (orange), Circadian (purple), Summaries (blue)
- Show data flow from raw → features → summaries
- Include example metric names (but not all 21)

**Why:** Shows what features are extracted and how they're organized into summaries

**Option B: Feature Table Schema** (Alternative)
```
Show a simplified table structure:

Per-Fly Features (fly_sleep_mean.csv):
┌─────────┬──────────────┬─────────────┬──────────────┬────────────┐
│ fly_id  │ total_sleep  │ n_bouts     │ sleep_eff    │ amplitude  │
├─────────┼──────────────┼─────────────┼──────────────┼────────────┤
│ 5-ch7   │ 450 min      │ 12          │ 62%          │ 0.85       │
│ 5-ch8   │ 380 min      │ 18          │ 53%          │ 0.72       │
└─────────┴──────────────┴─────────────┴──────────────┴────────────┘
         ↓ (Group by Genotype, Sex, Treatment)
Group Summaries (group_sleep_summary.csv):
┌──────────┬──────────┬──────────────┬──────────────┬────────────┐
│ Genotype │ Sex      │ total_sleep  │ sleep_eff    │ amplitude  │
│          │          │ (mean ± SEM)  │ (mean ± SEM) │ (mean±SEM) │
├──────────┼──────────┼──────────────┼──────────────┼────────────┤
│ WT       │ Female   │ 420 ± 15     │ 58 ± 2%      │ 0.78 ± 0.05│
│ Mutant   │ Female   │ 350 ± 20     │ 48 ± 3%      │ 0.65 ± 0.08│
└──────────┴──────────┴──────────────┴──────────────┴────────────┘
```

**Visual Style:**
- Clean table format
- Show transformation: per-fly → group summaries
- Highlight key metrics (not all 21)
- Use arrows to show aggregation

**Why:** Shows concrete output structure - biologists can see what they get

**Option C: Feature Categories Diagram** (Simplest)
```
Show feature categories as boxes:

[Feature Extraction]
    │
    ├─→ [Sleep Architecture]
    │   • Duration, Timing, Fragmentation
    │   • Efficiency, Latency, Transitions
    │
    └─→ [Circadian Rhythm]
        • Strength, Timing, Baseline

    ↓

[Automated Summaries]
    │
    ├─→ [Per-Fly] → Individual phenotypes
    └─→ [Group-Level] → Statistical comparisons
```

**Visual Style:**
- Simple boxes with categories
- Minimal text
- Clear hierarchy

**Why:** Simplest option, focuses on categories rather than specific metrics

**Recommendation:** Use **Option A (Feature Hierarchy Diagram)** - it shows both what features are extracted AND how they're organized into summaries, which directly addresses points 3 and 4.

---

## **SLIDE 4: Current Capabilities - What Questions Can We Answer?**

**MUST HAVE IMAGE:** Side-by-side actograms

**Generate from your pipeline:**
- Use `generate_actogram.py` to create two actograms
- Left: Wild-type or control fly
- Right: Mutant or treated fly
- Annotate with arrows/boxes highlighting differences:
  - "Fragmented sleep" (point to many short bars)
  - "Phase shift" (point to activity peak at different time)
  - "Reduced amplitude" (point to smaller bars overall)

**Alternative/Additional:**
- **Feature comparison bar chart** (from `create_features.py` output)
  - Show 4-5 key metrics side-by-side
  - Use your `group_sleep_summary.csv` data
  - Highlight significant differences with asterisks

**Why:** Actograms are the gold standard visualization in sleep research - biologists immediately understand them

---

## **SLIDE 5: Example - Biological Discovery Enabled by the Pipeline**

**MUST HAVE IMAGE:** Before/After comparison table

**Create this:**
| Metric | Traditional Analysis | Pipeline Analysis | Biological Insight |
|--------|---------------------|------------------|-------------------|
| Total Sleep | No difference | No difference | - |
| Sleep Efficiency | ❌ Not measured | ↓ 30% | Sleep quality disrupted |
| Dark Sleep | ❌ Not measured | ↓ 40% | Nighttime sleep affected |
| P_wake | ❌ Not measured | ↑ 2x | Sleep instability |
| Circadian Amplitude | ❌ Not measured | ↓ 25% | Rhythm weakened |

**Visual Style:**
- Use checkmarks/X marks for "measured vs not measured"
- Color-code: Red for decreases, Green for increases
- Keep it simple and readable

**Additional Image (optional):**
- **Actogram comparison** showing the same example fly
- Why: Visual confirmation of the metrics

**Why:** Shows concrete value - what we would have missed

---

## **SLIDE 6: The Next Step - Machine Learning for Pattern Discovery**

**Image Options:**
1. **Conceptual diagram: High-dimensional space** (create simple graphic)
   - Show: 21 features = 21 dimensions
   - Visual: 3D cube with dots (representing flies) scattered inside
   - Label: "How do we find patterns in 21 dimensions?"
   - Why: Makes the dimensionality problem tangible

2. **OR: Simple PCA visualization** (if you have example data)
   - 2D scatter plot showing first two principal components
   - Color-code by genotype or treatment
   - Why: Shows what PCA does visually

**Recommendation:** Use the conceptual diagram - it's accessible and sets up the ML need

---

## **SLIDE 7: ML-Enabled Biological Discoveries (Future Vision)**

**MUST HAVE IMAGE:** Hypothetical UMAP/clustering plot

**Create this (even if hypothetical):**
- 2D scatter plot with colored clusters
- Each cluster labeled: "Fragmented sleep", "Phase-shifted", "Both disrupted"
- Color-code by behavioral phenotype (not genotype)
- Add annotation: "Flies cluster by behavior, revealing hidden subtypes"

**Visual Style:**
- Use distinct colors for each cluster
- Add legend explaining what each cluster represents
- Keep it clean and professional

**Why:** Shows the power of ML pattern discovery in a way biologists can understand

**Additional Image (optional):**
- **Feature importance bar chart** (hypothetical Random Forest output)
  - Shows which features best distinguish groups
  - Why: Demonstrates feature selection capability

---

## **SLIDE 8: Impact for the Lab - Beyond This Project**

**Image Options:**
1. **Timeline comparison** (simple graphic)
   - Before: Manual analysis timeline (2-3 days)
   - After: Pipeline timeline (30 minutes)
   - Visual: Two horizontal bars, different lengths
   - Why: Quantifies time savings visually

2. **OR: Project applicability diagram**
   - Center: "Pipeline"
   - Branches to: "Sleep mutants", "Alzheimer's flies", "Drug screens", "Aging studies"
   - Why: Shows extensibility

**Recommendation:** Use the timeline - it's concrete and impressive

---

## **SLIDE 9: Technical Decisions - Biological Rationale**

**No image needed** - This slide is about explaining rationale, text is sufficient

**Optional:** Simple diagram showing:
- "1-minute bins" → "Sleep bouts (minutes)"
- "1-hour bins" → "Circadian rhythms (hours)"
- Visual: Two time scales side-by-side

---

## **SLIDE 10: Closing - From Computation to Discovery**

**Image Options:**
1. **Circular diagram** (create simple graphic)
   - Center: "Biological Questions"
   - Outer ring: "Computational Tools" → "Feature Extraction" → "Pattern Discovery" → "Biological Insights" → back to "Biological Questions"
   - Why: Shows the iterative cycle of discovery

2. **OR: Summary visualization**
   - Three panels: "What we built" → "What we can do" → "What's next"
   - Simple icons for each
   - Why: Reinforces the narrative arc

**Recommendation:** Use the circular diagram - emphasizes that computation serves biology

---

## **Additional Image Suggestions (Use Throughout)**

### **1. Actogram Examples** (Generate from your data)
- **Best use:** Slide 4, Slide 5
- **How to generate:** Use `generate_actogram.py` for representative flies
- **Tips:**
  - Choose flies that show clear differences
  - Generate high-resolution (300 DPI) for presentation
  - Add annotations in PowerPoint/Illustrator after generation

### **2. Feature Comparison Charts** (From your output files)
- **Best use:** Slide 4, Slide 5
- **Data source:** `group_sleep_summary.csv` or `fly_sleep_mean.csv`
- **Create:** Bar charts comparing 4-5 key metrics between groups
- **Tips:**
  - Use error bars (SEM)
  - Color-code by significance
  - Keep it simple - don't show all 21 features at once

### **3. Circadian Rhythm Curves** (Generate from your data)
- **Best use:** Slide 4 (circadian section)
- **How to create:** Plot mean activity vs. ZT hour for a group
- **Visual:** Line plot with shaded error bands
- **Annotate:** Mark peak (acrophase), amplitude, mesor

### **4. Conceptual Diagrams** (Create in PowerPoint/Illustrator)
- **Pipeline flowchart** (Slide 3)
- **Feature hierarchy** (Slide 3, points 3 & 4)
- **High-dimensional space** (Slide 6)
- **ML clustering** (Slide 7)
- **Discovery cycle** (Slide 10)

---

## **Image Creation Tips**

### **For Actograms:**
```bash
# Generate actograms for presentation
python generate_actogram.py 5 16  # Control fly
python generate_actogram.py 5 17  # Mutant fly (if available)
# Then annotate in PowerPoint/Illustrator
```

### **For Feature Charts:**
- Use your `group_sleep_summary.csv` data
- Create in Python (matplotlib) or Excel
- Export as high-resolution PNG (300 DPI minimum)
- Keep color scheme consistent across all charts

### **For Conceptual Diagrams:**
- Use PowerPoint SmartArt or simple shapes
- Keep it minimal - less is more
- Use consistent color scheme
- Test readability from back of room

### **General Guidelines:**
- **Resolution:** Minimum 300 DPI for printed handouts, 150 DPI for screen
- **File format:** PNG for photos/diagrams, SVG for simple graphics (if possible)
- **Color:** Use colorblind-friendly palettes (avoid red/green alone)
- **Text:** Make sure all text is readable at presentation size
- **Consistency:** Use same style/fonts across all images

---

## **Priority Images (Must Have)**

1. **Slide 3:** Pipeline flowchart
2. **Slide 3 (points 3 & 4):** Feature hierarchy diagram (Option A)
3. **Slide 4:** Side-by-side actograms
4. **Slide 5:** Before/After comparison table
5. **Slide 7:** Hypothetical UMAP/clustering plot

These five images are the most important for telling your story visually.

---

## **Quick Image Checklist**

- [ ] Pipeline flowchart (Slide 3)
- [ ] Feature hierarchy diagram (Slide 3, points 3 & 4) ⭐ NEW
- [ ] Actogram comparison (Slide 4)
- [ ] Feature comparison chart (Slide 4 or 5)
- [ ] Before/After table (Slide 5)
- [ ] High-dimensional space diagram (Slide 6)
- [ ] Hypothetical UMAP plot (Slide 7)
- [ ] Timeline comparison (Slide 8)
- [ ] Discovery cycle diagram (Slide 10)
