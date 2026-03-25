# Questions for Professor / Paper Author

## Purpose of This Note

This note separates three things clearly:

1. What is directly supported by `REFERENCE_MODEL_FINAL.md`
2. What was assumed during implementation because the paper does not fully specify it
3. What should be clarified with the professor or original paper author before treating the current code as a faithful research implementation

This note is intentionally critical. It does not assume that the current implementation choices are correct just because they are working in code.

## 1. State Scaling vs Reward Design

These are different issues.

### State scaling

The environment produces a raw state with measurable quantities:

- estimated CFO
- synchronization mismatch
- SINR
- pilot-domain SNR
- estimated channel power
- previous pilot ratio
- previous pilot power fraction

Before passing this state to the neural network, the implementation rescales the values. For example:

- CFO is divided by a reference scale
- some positive quantities use `log(1 + x)`
- all features are clipped to a fixed range

This does **not** change the physical meaning of the measurements. It only makes neural-network training numerically easier.

### Reward design

The reward defines what the agent is trying to optimize:

`reward = lambda_1 * rate - lambda_2 * sync_metric - lambda_3 * sensing_metric`

This is a separate design issue. If the reward is poorly balanced, the agent can learn the wrong behavior even when the state scaling is good.

### Key conclusion

State scaling is **not** a fix for reward-design problems.

It helps optimization stability, but it does not solve the question:

"Are the reward terms weighted in a scientifically justified way?"

## 2. Current Reward Design in the Code

The current implementation uses:

- `lambda_1 = 1.0` for rate
- `lambda_2 = 1.0` for synchronization mismatch
- `lambda_3 = 0.5` for sensing metric

These were chosen as practical placeholder values for a working reference implementation.

They are **not** normalized and they do **not** need to sum to 1.

However, that does **not** mean they are automatically good research choices.

What matters is whether the three reward terms are on compatible numerical scales.

If typical values are approximately:

- `rate ~ 2 to 3`
- `sync mismatch ~ 0.1`
- `sensing metric ~ 0.2 to 0.3`

then the reward is dominated by the rate term. That may or may not match the paper's intended tradeoff.

## 3. Can This Be Used on Actual Hardware Tomorrow?

Short answer: **no, not directly**.

### What is hardware-plausible

The paper's overall idea is hardware-plausible because it says the controller should use measurable quantities only.

That part is reasonable.

### What is not hardware-ready in the current implementation

The current code is still a simulator and proof-of-concept implementation. It includes simplifying assumptions:

- block-fading scalar channel per frame
- mean-reverting CFO process chosen for stable simulation
- synthetic noise and fading
- a sensing metric implemented as a measurable proxy chosen during implementation
- no SDR or real radio interface
- no real-time inference or frame formatting pipeline

### Key conclusion

The current system is useful as a simulation implementation of the paper structure.

It is **not** the same thing as a deployable hardware system.

To move toward hardware, the professor or author needs to confirm:

- which measurable sensing metric should be used in practice
- whether the pilot repetition assumption is acceptable
- whether the frame-by-frame control loop is realistic
- what physical operating regime the system is supposed to target

## 4. What Came Directly From the Paper

The following parts are directly aligned with the final reference model:

- adaptive action variables are pilot ratio `rho_t` and pilot power fraction `beta_t`
- discrete action set:
  - `rho in {0.1, 0.2, 0.3, 0.4}`
  - `beta in {0.2, 0.4, 0.6}`
- CFO is estimated from repeated pilots across adjacent OFDM symbols
- synchronization quality uses an observable residual mismatch metric
- the state uses measurable quantities from the previous frame
- the agent is implemented as a DQN with replay memory and target network
- the reward uses rate, synchronization quality, and sensing quality terms

## 5. What Was Assumed During Implementation

The paper gives the framework, but it does not uniquely determine every numeric setup. The following values were chosen during implementation and should be treated as assumptions unless the professor confirms them.

### Environment parameters currently used

- number of subcarriers: `K = 40`
- OFDM symbols per frame: `N = 8`
- average power: `P = 1.0`
- noise variance: `sigma^2 = 0.05`
- OFDM symbol duration: `T_s = 66.666e-6`
- subcarrier spacing: `15 kHz`
- CFO mean-reversion coefficient: `0.95`
- CFO innovation standard deviation: `5 Hz`
- channel correlation coefficient: `0.98`
- episode length: `64` frames

### Sensing setup currently used

The paper says the sensing term should be observable, but it does not fully lock down a single implementation formula.

The current default code uses a measurable proxy:

- build a delay-correlation profile from pilot-domain channel estimates after CFO compensation
- define sensing quality by the sidelobe-to-peak ratio

This is implementable and observable, but it is still an implementation choice, not a confirmed paper-authorized formula.

The code also supports an optional CRLB mode for comparison with earlier paper drafts, but that mode should be treated as a comparison setting rather than the final-paper default.

### DQN training setup currently used

- replay capacity: `4000`
- batch size: `32`
- hidden layers: `(48, 48)`
- learning rate: `1e-3`
- discount factor: `0.98`
- epsilon-greedy from `1.0` down to `0.05`
- warmup steps: `128`
- target update interval: `100`

These are engineering choices, not paper-derived truths.

## 6. Exact Questions to Ask the Professor / Original Author

Below is the recommended question list. These should be asked explicitly, not assumed.

### A. Simulation parameter clarification

1. What exact values should I use for the environment parameters?
   Ask for:
   - number of subcarriers `K`
   - number of OFDM symbols `N`
   - average transmit power or operating SNR
   - noise variance
   - CFO drift parameters
   - channel fading model and temporal correlation
   - episode length and number of training episodes

2. Are these values meant to represent a specific practical wireless scenario, or are they only illustrative?

### B. Sensing metric clarification

3. What exact measurable sensing metric should I implement for `S_t`?

4. Should `S_t` represent:
   - delay estimation error
   - delay-correlation sharpness
   - a CRLB-like proxy
   - range error
   - some other observable sensing metric

5. If the paper intentionally leaves `S_t` open-ended, what specific sensing metric do you want used for the project?

### C. Reward clarification

6. Should the reward use raw terms directly, or should each term be normalized by a reference scale first?

7. How should `lambda_1`, `lambda_2`, and `lambda_3` be selected?
   Ask whether they should be:
   - chosen heuristically
   - tuned experimentally
   - derived from target operating constraints
   - set so the reward terms have comparable scale

8. Is the goal to maximize rate primarily, with synchronization and sensing as soft penalties, or should the three objectives be treated more evenly?

### D. Baseline and evaluation clarification

9. What baselines should I compare against?
   Recommended baseline candidates:
   - fixed pilot ratio and power allocation
   - random action selection
   - simple heuristic adaptation rule

10. Which figures or tables are expected in the final project report?
   Possible outputs:
   - reward vs training episode
   - achievable rate vs policy
   - synchronization metric vs policy
   - sensing metric vs policy
   - learned action distribution

### E. Hardware relevance clarification

11. Is this project expected to remain a simulation study, or should it be designed with hardware deployment in mind?

12. If hardware relevance matters, which assumptions are acceptable?
   Ask specifically about:
   - repeated pilot symbols across consecutive OFDM symbols
   - flat block fading within a frame
   - frame-level action updates
   - measurable sensing metric available in a real receiver

13. Is the DQN itself important, or is the main goal to demonstrate adaptive pilot/power control under the paper's model?

## 7. Suggested Short Version to Send

If a shorter message is needed, the following version can be sent:

---

I have implemented the paper framework in simulation, but I need clarification on the parts that are not numerically specified in the manuscript.

1. What exact environment parameters should I use for `K`, `N`, SNR/noise variance, CFO drift, channel correlation, and episode length?
2. What exact observable sensing metric should be used for `S_t`?
3. Should the reward terms be used raw, or normalized before applying `lambda_1`, `lambda_2`, and `lambda_3`?
4. How should the reward weights be chosen?
5. What baselines should I compare the DQN against?
6. Is the expected outcome a simulation-only study, or should the design be kept hardware-plausible?

Also, if there is a specific numerical setup or benchmark configuration you want followed, please share it so I do not invent assumptions that deviate from the intended model.

---

## 8. Recommended Next Step After Clarification

Once the professor answers the questions above, the implementation should be updated in this order:

1. lock the environment parameters to the approved values
2. replace the sensing proxy if the professor specifies a different `S_t`
3. revise the reward if normalization or different weights are required
4. add baseline policies
5. run multi-seed experiments and compare results

## 9. Final Critical Summary

The current implementation is best described as:

- structurally faithful to the paper's control framework
- numerically plausible
- suitable for simulation prototyping
- not yet validated as the exact intended experimental setup

The biggest unresolved issues are:

- the exact sensing metric
- the exact environment parameters
- the scientific justification for reward weighting
- the expected evaluation baselines
