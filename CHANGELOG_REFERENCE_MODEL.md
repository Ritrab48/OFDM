# Changelog: Revisions to Reference Model

This document details all the structural, mathematical, and logical corrections made to the original `REFERENCE_MODEL_FOR_GITHUB.md` to produce `REFERENCE_MODEL_FOR_GITHUB_REVISED.md`.

## 1. Resolved Causality Paradox in the MDP State Space
* **Original Issue:** The Markov Decision Process (MDP) state space $s_t$ required the agent to observe the current frame's CFO estimate $\hat{\epsilon}_t$ and channel magnitude $|H_t|^2$ to select the current action $(\rho_t, \beta_t)$. However, computing these observations requires processing the received signal, which is generated *after* the action is applied to format the transmitted signal. This created a "chicken-and-egg" temporal paradox.
* **Resolution:** Shifted the state definition to use strict delayed observations. The state space now utilizes measurements from the previously completed frame: $s_t = [\,\hat{\epsilon}_{t-1}, \dots, |\hat{H}_{t-1}|^2, \dots\,]$.

## 2. Removed Unobservable Quantities from the State Space
* **Original Issue:** The state space included the true squared residual CFO error, $\tilde{\epsilon}_t^2$. By definition, the true CFO is unmeasurable at the receiver, making its exact error unobservable in a real-world, online execution context.
* **Resolution:** Replaced the unobservable instantaneous true error with the theoretically computable Cramér-Rao Lower Bound ($\mathrm{CRLB}$) of the CFO estimate: $\mathrm{CRLB}(\hat{\epsilon}_{t-1})$. This acts as a proxy that informs the agent of its current estimation confidence based on its previous actions.

## 3. Standardized Abstraction Levels in the Reward Function
* **Original Issue:** The reward function $r_t$ (Equation 15) mixed an instantaneous, noise-dependent simulated error ($\tilde{\epsilon}_t^2$) with a theoretical expected lower bound ($\mathrm{Var}(\hat{\tau}_t)$). This inconsistency can cause severe variance and instability during Reinforcement Learning training.
* **Resolution:** Unified the reward function to operate entirely on expected theoretical bounds. It now penalizes based on $\mathrm{CRLB}(\hat{\epsilon}_t)$ and $\mathrm{CRLB}(\hat{\tau}_t)$ instead of specific error realizations. The text and equations were updated to explicitly use the `CRLB()` operator to resolve ambiguity regarding whether the metric was an actual simulated variance or a theoretical bound.

## 4. Corrected Physical Modeling of Inter-Carrier Interference (ICI)
* **Original Issue:** The ICI power equation (Equation 10) scaled the interference with total transmit power $P_t$, but omitted the channel fading gain $|H_t|^2$. Meanwhile, the desired signal correctly included channel fading. This implied self-interference was magically immune to the wireless channel.
* **Resolution:** Updated Equation 10 to include the channel gain: $\sigma^2_{\mathrm{ICI},t} \approx C \cdot \tilde{\epsilon}_t^2 P |H_t|^2$.

## 5. Corrected Mathematical Bias in the CFO Estimator
* **Original Issue:** The least-squares CFO estimator (Equation 7) computed phase differences between consecutive OFDM symbols. A frame with $N$ symbols has exactly $N-1$ consecutive pairs. However, the estimator (and its variance bound in Equation 8) incorrectly averaged the result by dividing by $N$.
* **Resolution:** Changed the denominator in Equation 7 and Equation 8 from $N$ to $N-1$. Added explanatory text to clarify this mathematical structure.

## 6. Corrected Sensing Time-Bandwidth Dependency
* **Original Issue:** The sensing delay estimation bound (Equation 13) depended on pilot power ($\beta_t$) but completely ignored the pilot density ratio ($\rho_t$). This contradicted the paper's claim that both variables jointly affect sensing.
* **Resolution:** Updated the sensing bound equation to explicitly include $\rho_t K N$ in the denominator, properly reflecting that sensing resolution depends on the effective time-bandwidth product (the number of subcarriers allocated to pilots).

## 7. Fixed Ambiguous Transmit Power Dynamics ($P_t \to P$)
* **Original Issue:** Transmit power was defined as a time-varying variable $P_t$, but no transition dynamics were provided. If it varied randomly, the environment was open-loop; if it was constant, the $t$ index was misleading. Furthermore, this caused the state space variable $\mathrm{SNR}_t = P_t/\sigma^2$ to be potentially static and useless.
* **Resolution:** 
  * Replaced all instances of $P_t$ with a constant $P$.
  * Clarified the definition as "constant average transmit power per OFDM symbol."
  * Removed $\mathrm{SNR}_{t-1}$ from the dynamic state space vector entirely, as it is now a static environmental constant that the neural network does not need to observe at every timestep.

## 8. Clarified Channel Model Granularity
* **Original Issue:** The text and equations jumped back and forth between a frequency-selective subcarrier-level channel $H_t[k,n]$ and a scalar flat-fading block channel $H_t$ without explanation.
* **Resolution:** Updated the received signal equation (Equation 3) to use the scalar $H_t$ and explicitly stated the assumption: *"we assume a block-fading channel where the channel response $H_t$ is approximately flat across subcarriers and time-invariant over the frame duration."*

## 9. Tempered Theoretical Convergence Claims
* **Original Issue:** Section 9 made overly strong theoretical guarantees about the convergence of Deep Q-Networks (DQN) with nonlinear function approximation. Standard DQNs lack formal mathematical convergence guarantees in continuous environments.
* **Resolution:** Rewrote Section 9 to accurately reflect that while formal guarantees do not exist, the use of experience replay, target networks, and theoretically bounded reward functions *empirically stabilize* the learning process.