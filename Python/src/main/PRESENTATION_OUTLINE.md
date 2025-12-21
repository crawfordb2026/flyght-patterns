# Drosophila Sleep Analysis Pipeline: Presentation Outline
## From Data to Discovery: Computational Tools for Sleep Biology

---

## **SLIDE 1: Opening - The Biological Question**

**Hook:** "When we screen sleep mutants or test treatments, we're asking: *How does this change sleep behavior?* But our DAM system generates millions of data points per experiment. How do we extract biological meaning from this complexity?"

**Key Points:**
- Sleep is multidimensional: duration, timing, fragmentation, circadian coupling
- Traditional analysis looks at one metric at a time (e.g., "total sleep")
- Real sleep phenotypes are complex combinations of behaviors
- **Our goal:** Build tools that capture the full sleep phenotype automatically

**Biological Motivation:**
- Identify subtle sleep mutants that single-metric analysis misses
- Understand how treatments affect sleep architecture (not just duration)
- Discover behavioral signatures that predict health outcomes
- Enable high-throughput screening of genetic variants

---

## **SLIDE 2: The Challenge - Why We Need Computational Tools**

**Key Points:**
- DAM system: 64 flies × 7 days × 1,440 minutes = 645,120 data points per experiment
- Manual analysis is time-consuming and error-prone
- Single-metric comparisons miss complex phenotypes
- Need reproducible, standardized analysis across experiments

**Biological Rationale:**
- Sleep fragmentation (many short bouts) vs. consolidated sleep (few long bouts) have different biological implications
- Circadian disruption can affect sleep even when total sleep time is normal
- Individual variation matters - some flies may show strong effects while others don't
- We need to capture these nuances to understand genotype-phenotype relationships

---

## **SLIDE 3: The 4-Step Pipeline - From Raw Data to ML-Ready Features**

**Frame as:** "A computational pipeline that automatically processes raw DAM data into clean, ML-ready features"

**Step 1: Prepare Data + Health Report**
- Loads raw DAM files (Monitor*.txt) and metadata (details.txt)
- Calculates ZT (Zeitgeber Time), Phase (Light/Dark), Exp_Day
- Generates health report: identifies dead/unhealthy flies (24h inactivity, activity drops, etc.)

**Step 2: Remove Flies (Optional)**
- Remove flies based on health report (Dead, QC_Fail, Unhealthy)
- Or remove by genotype, sex, treatment, specific fly IDs
- Flexible criteria via command line or config

**Step 3: Create Feature Table**
- Extracts **sleep features**: daily metrics (total sleep, bouts, fragmentation, latency, etc.)
- Extracts **circadian features**: daily cosinor regression → per-fly means/SDs (Mesor, Amplitude, Phase)
- Merges into single `ML_features.csv` table

**Step 4: Clean Features**
- Removes flies with zero sleep/bouts/P_doze
- Removes IQR outliers per group
- Fixes NaN values
- Creates z-scored features (`ML_features_Z.csv`)

**Output:** Clean, normalized features ready for ML analysis

**Visual:** Flowchart: Raw Data → Step 1 → Step 2 (optional) → Step 3 → Step 4 → ML Features

---

## **SLIDE 3B: Health Report Decision Tree - How We Identify Dead/Unhealthy Flies**

**Frame as:** "The health report automatically classifies each fly's health status using a simple decision tree"

**Simplified Decision Tree:**

```
For each fly, each day:

    ┌─────────────────────────────────────┐
    │  Is the fly moving?                 │
    └──────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
  NO MOVEMENT (MT)      HAS MOVEMENT
   for the whole              |
    24 hr day?                │
        │                     │
        YES                   │
        │                     │
        ▼                     │
    ┌─────────┐               │
    │  DEAD   │               │
    └─────────┘               │
                              │
        ┌─────────────────────┘
        │
    ┌─────────────────────────────────────┐
    │  No movement for 12 hours           │
    │  AND no response to light change?   │
    └──────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        YES                   NO
        │                     │
        ▼                     │
    ┌─────────┐               │
    │  DEAD   │               │
    └─────────┘               │
                              │
        ┌─────────────────────┘
        │
    ┌─────────────────────────────────────┐
    │  Activity dropped below 20%         │
    │  of normal (reference day)?         │
    └──────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        YES                   NO
        │                     │
        ▼                     │
    ┌─────────┐               │
    │  DEAD   │               │
    └─────────┘               │
                              │
        ┌─────────────────────┘
        │
    ┌─────────────────────────────────────┐
    │  Very low activity OR too much      │
    │  sleep AND no response to light?    │
    └──────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        YES                   NO
        │                     │
        ▼                     │
    ┌─────────────┐           │
    │  UNHEALTHY  │           │
    └─────────────┘           │
                              │
        ┌─────────────────────┘
        │
    ┌─────────────────────────────────────┐
    │  More than 10% of data missing?     │
    └──────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        YES                   NO
        │                     │
        ▼                     ▼
    ┌──────────┐         ┌─────────┐
    │ QC_FAIL  │         │  ALIVE  │
    └──────────┘         └─────────┘
```

**In Simple Terms:**

1. **DEAD** = No movement for 24 hours, OR no movement for 12 hours + no response to light, OR activity dropped way down (<20% of normal)

2. **UNHEALTHY** = Very low activity or too much sleep + no response to light, OR activity dropped moderately (<50% of normal)

3. **QC_FAIL** = Too much missing data (>10%)

4. **ALIVE** = Everything else (healthy flies)

**Key Insight:** The "startle test" (response to light changes) distinguishes sleeping flies from dead flies - even sleeping flies wake up when lights change!

**Visual:** Use the decision tree diagram above, or create a flowchart-style visualization

---

## **SLIDE 4: Feature Extraction - What We Measure**

**Frame as:** "The pipeline extracts 25+ features per fly, capturing the full sleep phenotype"

**Sleep Architecture Features (17 metrics):**
- **Duration:** Total sleep, day sleep, night sleep
- **Bouts:** Total bouts, day/night bouts, mean/max bout lengths
- **Fragmentation:** Fragmented bouts per hour, per minute of sleep
- **Quality:** Sleep efficiency, WASO (Wake After Sleep Onset), sleep latency
- **Transitions:** P_wake (probability of waking), P_doze (probability of falling asleep)
- **Wake:** Mean wake bout length

**Circadian Rhythm Features (8 metrics):**
- **Per-fly means:** Mesor, Amplitude, Phase
- **Per-fly SDs:** Mesor_sd, Amp_sd, Phase_sd
- **Group-level:** Group Mesor, Amplitude, Acrophase, p-value

**Biological Value:**
- Comprehensive but interpretable
- Based on established Drosophila sleep research standards
- Captures both sleep architecture and circadian coupling

---

## **SLIDE 5: ML Analysis Tools - Pattern Discovery**

**Frame as:** "Once we have features, how do we find patterns across 25+ dimensions?"

**What We've Built:**

**1. PCA Analysis** ✅
- Finds main axes of variation in sleep phenotypes
- Statistical comparisons: Which genotypes differ? (ANOVA/Kruskal-Wallis with auto-selection)
- Outputs: PC1 vs PC2 plots, genotype signature heatmaps, effect size rankings

**2. UMAP + DBSCAN Clustering** ✅
- Non-linear dimensionality reduction (preserves local structure)
- Automatically discovers behavioral clusters (automated eps detection)
- Cluster analysis: Which genotypes cluster together? What defines each cluster?
- Genotype comparisons within and across clusters
- Effect sizes (Cliff's Delta) for all genotype pairs
- Outputs: UMAP plots, cluster assignments, cluster signatures, effect size heatmaps

**3. Sex Difference Analysis** ✅
- Analyzes sex distribution within clusters (Fisher's exact test)
- Genotype-specific sex comparisons (configurable: any genotype or "All")
- Outputs: Sex × cluster counts, statistical tests, sex comparison heatmaps

**Key Message:** These tools discover patterns we can then validate experimentally

---

## **SLIDE 6: What We Can Discover - Real Capabilities**

**Frame as:** "Here's what these tools enable us to discover right now"

**1. Phenotype Clustering**
- UMAP + DBSCAN automatically finds behavioral groups
- Example: "Cluster 1: Fragmented sleep, normal circadian. Cluster 2: Normal sleep, phase-shifted"
- Biological value: Identifies distinct behavioral subtypes within genotypes

**2. Genotype Comparisons**
- PCA identifies which features distinguish genotypes
- Effect sizes (Cliff's Delta) show magnitude of differences
- Statistical tests with FDR correction
- Biological value: Prioritizes which mutants to follow up

**3. Sex Differences**
- Analyzes sex distribution within clusters
- Genotype-specific sex comparisons (e.g., Rye males vs females)
- Biological value: Identifies sex-specific phenotypes

**4. Pattern Discovery**
- Which features co-vary? (PCA loadings)
- What defines each behavioral cluster? (Cluster signatures)
- Biological value: Reveals relationships between sleep metrics

**Visual:** Show actual UMAP plot and PCA plot from your data

---

## **SLIDE 7: Example Workflow - From Data to Discovery**

**Tell a story:** "Here's how the complete pipeline works from start to finish"

**Example:**
1. **Run Pipeline:**
   - Step 1: `python 1-prepare_data_and_health.py` → Health report shows 7 dead flies
   - Step 2: `python 2-remove_flies.py --statuses Dead` → Removes dead flies
   - Step 3: `python 3-create_feature_table.py` → Creates ML_features.csv (50 flies, 25 features)
   - Step 4: `python 4-clean_ml_features.py` → Creates ML_features_Z.csv (48 flies after cleaning)

2. **Run Analysis:**
   - `python pca_analysis.py` → Finds 3 features that distinguish genotypes (p < 0.05)
   - `python umap_dbscan_analysis.py` → Discovers 2 behavioral clusters
   - `python sexdiff_analysis.py --genotype Rye` → Shows Rye males differ from females in 5 features

3. **Biological Insight:**
   - "Rye genotype shows sex-specific sleep fragmentation"
   - "Two distinct behavioral subtypes exist within the mutant group"
   - "Sleep efficiency and circadian amplitude are the strongest predictors of genotype"

**Key Point:** Complete pipeline from raw data to biological insights in ~1 hour

---

## **SLIDE 8: Impact for the Lab**

**Key Points:**

1. **Time Savings**
   - Manual analysis: ~2-3 days per experiment
   - Pipeline: ~30 minutes per experiment
   - **Enables:** More experiments, faster iteration

2. **Reproducibility**
   - Standardized 4-step pipeline across all experiments
   - No manual errors or subjective decisions
   - **Enables:** Meta-analyses, comparisons across studies

3. **Extensibility**
   - Same pipeline works for: Alzheimer's flies, drug screens, aging studies, any DAM experiment
   - **Enables:** Consistent analysis across all lab projects

4. **New Capabilities**
   - Can now ask questions we couldn't before (complex phenotypes, pattern discovery)
   - ML tools ready to use immediately
   - **Enables:** More sophisticated biological questions

**Concrete Examples:**
- "We can now re-analyze old experiments with the same pipeline"
- "The pipeline is ready for the Alzheimer's fly project"
- "ML analysis helps prioritize which mutants to follow up"

---

## **SLIDE 9: Technical Decisions - Biological Rationale**

**Frame as:** "Every technical choice was made to answer a biological question"

1. **1-minute bins for sleep, 1-hour bins for circadian**
   - Biological: Sleep bouts are minutes-long; circadian rhythms are hours-long
   - Rationale: Match the time scale to the biological process

2. **5-minute sleep threshold**
   - Biological: Standard in Drosophila sleep research (Hendricks et al., 2000)
   - Rationale: Distinguishes sleep from brief inactivity

3. **Health report with multiple criteria**
   - Biological: Death isn't always obvious; need multiple indicators
   - Rationale: 24h inactivity OR activity drop OR no startle response

4. **Daily cosinor then aggregate (not single cosinor per fly)**
   - Biological: Captures day-to-day variation in circadian rhythms
   - Rationale: More robust than single cosinor fit

5. **Z-scoring for ML analysis**
   - Biological: Features on different scales (minutes vs. probabilities vs. hours)
   - Rationale: Normalize so all features contribute equally to ML

**Key Message:** The pipeline is designed by biologists, for biologists

---

## **SLIDE 10: Closing - Ready to Use**

**Reconnect to opening:** "We started by asking: *How do we extract biological meaning from complex sleep data?*"

**What we've accomplished:**
- ✅ **4-step pipeline:** Data prep → Fly removal → Feature extraction → Cleaning
- ✅ **25+ features per fly:** Sleep architecture + circadian rhythms
- ✅ **3 ML analysis tools:** PCA, UMAP clustering, sex difference analysis
- ✅ **Standardized, reproducible:** Same analysis across all experiments
- ✅ **Ready to use:** All scripts tested and working

**What's next:**
- Apply these tools to your experiments
- Discover behavioral signatures in existing data
- Screen genetic variants at scale
- Identify predictive biomarkers

**Final Message:** "Computational tools don't replace biological insight - they amplify it. The pipeline is ready to use now."

**Call to Action:** "I'm excited to work with you all to apply these tools to your projects. What sleep questions should we tackle next?"

---

## **APPENDIX: Potential Questions & Answers**

### **Q: "Why not just use existing software?"**
**A:** "Existing tools (like ClockLab) are great but focus on single metrics. Our pipeline extracts 25+ features simultaneously, includes automated health assessment, and has ML analysis tools built in. Plus, it's open-source and customizable for our specific needs."

### **Q: "How do we know the features are biologically meaningful?"**
**A:** "All metrics are based on established Drosophila sleep research. The 5-minute sleep threshold, sleep efficiency calculation, and circadian parameters all follow published standards. We're quantifying what sleep biologists already measure, just automatically."

### **Q: "What if ML finds patterns that don't make biological sense?"**
**A:** "That's actually valuable! Unexpected patterns are hypotheses to test. ML is a discovery tool - we validate findings with follow-up experiments. It's hypothesis generation, not hypothesis replacement."

### **Q: "How long until we can use this for our experiments?"**
**A:** "Everything is ready now! The 4-step pipeline extracts features, and the 3 ML analysis scripts are ready to run. We can start analyzing existing data immediately."

### **Q: "Can this work with other behavioral assays?"**
**A:** "The core approach (feature extraction → pattern discovery) applies to any time-series behavioral data. The pipeline is designed to be extensible for other DAM-based experiments."

### **Q: "What's the learning curve for using these tools?"**
**A:** "For the pipeline: minimal - just run 4 scripts in order. For ML analysis: I'll provide templates and work with you to interpret results. The goal is to make these tools accessible, not to require everyone to become a programmer."

---

## **Visualization Recommendations**

### **Visualization 1: Pipeline Flowchart**
- **What:** Diagram showing: Raw DAM Files → Step 1 → Step 2 (optional) → Step 3 → Step 4 → ML Features
- **Why:** Shows the complete workflow
- **Annotate:** Highlight key outputs at each step

### **Visualization 2: Feature Comparison**
- **What:** Bar chart comparing 4-5 key metrics between genotypes
- **Why:** Shows the pipeline extracts multiple dimensions of difference
- **Annotate:** "While total sleep is similar, sleep efficiency and bout count reveal significant differences"

### **Visualization 3: UMAP Cluster Plot**
- **What:** 2D scatter plot showing fly clusters from UMAP
- **Why:** Demonstrates ML's pattern discovery capability
- **Annotate:** "Flies cluster by behavioral phenotype, not just genotype"

### **Visualization 4: PCA Plot**
- **What:** PC1 vs PC2 scatter plot colored by genotype
- **Why:** Shows main axes of variation
- **Annotate:** "PC1 captures sleep duration differences, PC2 captures fragmentation"

---

## **Presentation Tips**

1. **Start with biology, end with biology** - Frame everything in terms of biological questions and discoveries

2. **Use analogies** - "PCA is like finding the main axes of variation in a cloud of data points"

3. **Show, don't tell** - Use actual examples from your data when possible

4. **Acknowledge limitations** - "This is a tool, not a replacement for biological intuition"

5. **Emphasize collaboration** - "I built the tools, but you all will use them to make discoveries"

6. **Keep it conversational** - Avoid jargon, explain technical terms when needed

7. **End with excitement** - "I'm excited to see what we discover when we apply these tools to our sleep data"
