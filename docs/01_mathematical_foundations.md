# Mathematical Foundations & Operations Research

This document outlines the mathematical models, stochastic processes, and optimization algorithms governing AnimalLens.

---

## 1. Spatial Graph & Kinematic Formulations

In multi-animal tracking environments (e.g. aquaculture tanks, livestock enclosures), the spatial configuration at time $t$ is modeled as a dynamic directed interaction graph:

$$G(t) = (V(t), E(t))$$

Where:
* $V(t) = \{v_1, v_2, \dots, v_n\}$ represents the set of tracked individuals. Each node $v_i = (\vec{p}_i(t), \vec{v}_i(t), \theta_i(t))$ is defined by spatial position $\vec{p}_i = (x_i, y_i) \in [0, 1]^2$, velocity vector $\vec{v}_i = \frac{d\vec{p}_i}{dt}$, and heading angle $\theta_i$.
* $E(t) = \{e_{ij}(t)\}$ represents directed pairwise interaction edges weighted by relative spatial proximity, relative velocity, and angular approach alignment:

$$w_{ij}(t) = \exp\left(-\frac{\|\vec{p}_i(t) - \vec{p}_j(t)\|^2}{2\sigma_d^2}\right) \cdot \max\left(0, \frac{\vec{v}_i(t) \cdot (\vec{p}_j(t) - \vec{p}_i(t))}{\|\vec{v}_i(t)\| \|\vec{p}_j(t) - \vec{p}_i(t)\|}\right)$$

### Clark-Evans Spatial Dispersion Index ($R$)
To quantify whether animals exhibit territorial dispersion, random foraging, or crowding aggregation:

$$R = \frac{\bar{r}_A}{\bar{r}_E}$$

* $\bar{r}_A = \frac{1}{N} \sum_{i=1}^N d_{\text{min}}(i)$ is the observed mean Nearest Neighbor Distance (NND).
* $\bar{r}_E = \frac{1}{2\sqrt{\rho}}$ is the expected mean distance under Complete Spatial Randomness (CSR) with density $\rho = \frac{N}{A}$.
* **Interpretation**:
  * $R < 1$: Clustered / Aggregated distribution (e.g. schooling, mating clusters, sheltering).
  * $R \approx 1$: Poisson random spatial distribution.
  * $R > 1$: Uniform / Territorial dispersion (e.g. agonistic territory defense).

---

## 2. Stochastic Behavior Modeling (Markov Chains)

Animal behavior sequences are modeled as a discrete-time Markov chain on the finite state space $S = \{s_1, s_2, \dots, s_K\}$ (e.g. resting, normal movement, feeding, social interaction, aggression, reproduction, unknown).

### First-Order Transition Probability Matrix
The empirical transition probability $P_{ij}$ from behavior state $s_i$ to $s_j$ is given by:

$$P_{ij} = P(S_{t+1} = s_j \mid S_t = s_i) = \frac{N_{ij}}{\sum_{k=1}^K N_{ik}}$$

Where $N_{ij}$ is the observed frequency of transition from state $s_i$ to state $s_j$.

### Stationary State Distribution ($\pi$)
For an ergodic behavioral chain, the asymptotic equilibrium time budget satisfies:

$$\pi P = \pi \quad \text{subject to} \quad \sum_{i=1}^K \pi_i = 1$$

Deviations in the empirical transition matrix $\Delta P = \|P_{\text{observed}} - P_{\text{baseline}}\|$ serve as an early-warning signal for biological distress, water quality degradation, or disease onset.

---

## 3. Buffer & Latency Pareto Optimization

Real-time video inference on edge hardware operates under bounded memory and latency constraints. We solve the following optimization problem:

$$\max_{\text{FPS}, \Delta t_{\text{buffer}}, \text{Res}} \text{Accuracy}(\text{FPS}, \Delta t_{\text{buffer}}, \text{Res})$$

$$\text{subject to} \quad T_{\text{infer}} \le \frac{1}{\text{FPS}} \quad \text{and} \quad \text{Memory}(\text{FPS}, \Delta t_{\text{buffer}}, \text{Res}) \le M_{\text{max}}$$

AnimalLens solves this using an asynchronous `RollingVideoBuffer` ring buffer architecture that decouples frame ingestion from heavy temporal classification strides.
