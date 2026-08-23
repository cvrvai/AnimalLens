# Scientific Research Methodology

This document defines the quantitative ethology protocols, dataset partitioning standards, and statistical validation rules used in AnimalLens.

---

## 1. Quantitative Ethological Observation Protocols

AnimalLens formalizes observational ethology based on standardized methodology (*Altmann, 1974*; *Martin & Bateson, 2007*):

### A. Focal Animal Sampling
* **Protocol**: A single tracked individual (identified by persistent `track_id` or `animal_id`) is continuously monitored over observation duration $T$.
* **Metrics**: Total duration, frequency, and latency to specific behavioral states (e.g. foraging latency after feeding event).

### B. Instantaneous Scan Sampling
* **Protocol**: At fixed time intervals $\Delta t$ (e.g. every 30 seconds), the behavioral state of every individual in the arena is recorded simultaneously.
* **Metrics**: Cohort state proportion distribution:
  $$p_k(t) = \frac{N_k(t)}{N_{\text{total}}}$$

### C. All-Occurrences Sampling
* **Protocol**: High-value discrete social or reproductive events (mating, sparring, tail-flip escape, molting) are recorded continuously across all individuals.

---

## 2. Anti-Leakage Experimental Partitioning

A critical failure mode in video-based computer vision is **temporal data leakage**, where adjacent frames from the same video sequence appear in both training and test sets, artificially inflating benchmark scores.

### Strict Grouped Partitioning Rule
Datasets must be partitioned exclusively using grouped stratified splits:

$$\text{Group Identifier} = (\text{Session ID}, \text{Tank/Pen ID}, \text{Camera Angle}, \text{Biological Cohort})$$

* **Zero-Leakage Guarantee**: No two frames or temporal clips originating from the same recording session or tank may exist across train, validation, and test splits.

---

## 3. Statistical Significance & Inter-Annotator Reliability

Before training supervised classifiers, ground-truth video annotations are validated across multiple human ethologists.

### Cohen's Kappa ($\kappa$) and Fleiss' Kappa
Inter-rater agreement for categorical behavior labels is quantified as:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

* $P_o$: Observed proportional agreement among annotators.
* $P_e$: Hypothetical probability of chance agreement.
* **Standard**: Only behavior classes with $\kappa \ge 0.75$ (substantial agreement) are accepted into official benchmark datasets.

### Confidence Calibration & Expected Calibration Error (ECE)
To ensure confidence scores represent true posterior probabilities:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

Low-confidence predictions ($\text{conf} < \theta_{\text{uncertainty}}$) or high-entropy classifications are systematically routed to the **Active Learning Review Queue** under the `unknown` class.
