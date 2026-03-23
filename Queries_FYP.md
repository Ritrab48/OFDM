# Response to Queries
**March 22, 2026**

## 1 Causality- and Observability-Consistent MDP Formulation

The original MDP formulation introduces a causality inconsistency by defining the system state using variables that depend on the current received signal. Specifically, quantities such as the CFO estimate, channel gain, and instantaneous SNR are obtained from the received signal $Y_t$, which itself depends on the current action $(\rho_t, \beta_t)$. This violates the fundamental requirement that the state must be observable prior to action selection.

In addition, the original formulation incorporates the residual synchronization error $\tilde{\epsilon}_t = \epsilon_t - \hat{\epsilon}_t$ into the state representation. However, the true CFO $\epsilon_t$ is not observable at the receiver, and therefore $\tilde{\epsilon}_t$ cannot be computed in practice. This renders the inclusion of $\tilde{\epsilon}_t^2$ in the state inconsistent with the requirement that all state variables must be measurable.

To ensure both causal and observational consistency, the state must be constructed using only information available prior to the current decision and derived from measurable quantities. Therefore, we redefine the system state using delayed observations from the previous frame and replace the unobservable residual error with a measurable surrogate metric.

Specifically, we define an observable synchronization error metric based on pilot-domain residual mismatch:

$$ \mathcal{E}_t = \frac{1}{|P_t|N} \sum_{k \in P_t} \sum_{n=1}^{N-1} |Y_t[k, n+1] - Y_t[k, n]e^{j2\pi\hat{\epsilon}_t T_s}|^2 \quad (1) $$

This metric captures the residual phase inconsistency after CFO compensation and can be computed directly from received pilot symbols.

The corrected state definition is therefore given by:

$$ s_t = \left( \hat{\epsilon}_{t-1}, \mathcal{E}_{t-1}, \widehat{\text{SNR}}_{t-1}, |\hat{H}_{t-1}|^2, \rho_{t-1}, \beta_{t-1} \right) \quad (2) $$

Under this formulation, the decision process follows a causally valid sequence:

$$ s_t \to a_t \to Y_t \to s_{t+1} \quad (3) $$

where the next state $s_{t+1}$ is computed using measurements obtained from $Y_t$. This ensures that all state variables are both measurable and available at the time of decision-making, thereby satisfying the requirements of a valid Markov decision process under practical receiver constraints.

This modification preserves the tractability of the reinforcement learning framework while ensuring physical realizability. Furthermore, it aligns the formulation with a partially observable system interpretation, where the agent operates based on measurable statistics derived from the received signal rather than inaccessible ground-truth parameters.

## 2 Observability-Consistent State and Reward Formulation

The original formulation defines the residual synchronization error as $\tilde{\epsilon}_t = \epsilon_t - \hat{\epsilon}_t$ and incorporates $\tilde{\epsilon}_t^2$ into both the MDP state and reward function. However, this introduces a fundamental inconsistency. The true carrier frequency offset (CFO) $\epsilon_t$ is not directly observable at the receiver, and therefore the residual error $\tilde{\epsilon}_t$ cannot be computed in an online implementation.

This violates the requirement that both the state and reward must be functions of measurable quantities. Moreover, if $\tilde{\epsilon}_t$ were available, the receiver could directly recover the true CFO via $\epsilon_t = \hat{\epsilon}_t + \tilde{\epsilon}_t$, rendering the estimation process unnecessary. Therefore, the use of $\tilde{\epsilon}_t^2$ implicitly assumes access to oracle information and is not physically realizable.

To address this issue, we replace the unobservable residual error with a measurable surrogate that captures synchronization quality. Specifically, we define an observable error metric based on pilot-domain residual mismatch:

$$ \mathcal{E}_t = \frac{1}{|P_t|N} \sum_{k \in P_t} \sum_{n=1}^{N-1} |Y_t[k, n+1] - Y_t[k, n]e^{j2\pi\hat{\epsilon}_t T_s}|^2 \quad (4) $$

This metric quantifies the residual phase inconsistency after CFO compensation and can be computed directly from received pilot symbols.

The state is therefore redefined as:

$$ s_t = \left( \hat{\epsilon}_{t-1}, \mathcal{E}_{t-1}, \widehat{\text{SNR}}_{t-1}, |\hat{H}_{t-1}|^2, \rho_{t-1}, \beta_{t-1} \right) \quad (5) $$

Similarly, the reward function is modified to penalize the observable synchronization error:

$$ r_t = \lambda_1 R_t - \lambda_2 \mathcal{E}_t - \lambda_3 \text{Var}(\hat{\tau}_t) \quad (6) $$

This reformulation ensures that both the state and reward depend only on measurable quantities, thereby restoring physical consistency and enabling practical implementation. Furthermore, it aligns the reinforcement learning formulation with a partially observable system interpretation, where the agent operates based on estimated and observed statistics rather than inaccessible ground-truth parameters.

## 3 Consistent Receive Model and ICI Characterization

The original receive model represents the effect of carrier frequency offset (CFO) as a per-symbol phase rotation:

$$ Y_t[k, n] = H_t[k, n]X_t[k, n]e^{j2\pi\epsilon_t n T_s} + W_t[k, n] \quad (7) $$

This formulation captures the temporal phase rotation induced by CFO but does not explicitly account for the loss of orthogonality among subcarriers. In practical OFDM systems, CFO introduces inter-carrier interference (ICI) due to leakage of signal energy across subcarriers. A rigorous representation involves coupling across all subcarriers, leading to:

$$ Y_t[k] = \sum_{m=0}^{K-1} H_t[m]X_t[m]S(k - m, \epsilon_t) + W_t[k] \quad (8) $$

where $S(k - m, \epsilon_t)$ denotes the ICI coefficient. However, this exact model is analytically intractable for control design.

To maintain tractability, we adopt a widely used small-CFO approximation in which the aggregate ICI power is modeled as a function of the residual CFO. Under this approximation, and assuming a flat-fading channel within each OFDM symbol, the ICI power is expressed as:

$$ \sigma^2_{\text{ICI},t} \approx C \cdot \tilde{\epsilon}_t^2 \cdot (1 - \beta_t)P_t \cdot |H_t|^2 \quad (9) $$

where $C$ is a constant determined by OFDM system parameters.

This formulation ensures consistency with the physical interpretation that both the desired signal and the interference components propagate through the same channel.

The resulting effective signal-to-interference-plus-noise ratio (SINR) is given by:

$$ \text{SINR}_t = \frac{(1 - \beta_t)P_t |H_t|^2}{\sigma^2 + \sigma^2_{\text{ICI},t}} \quad (10) $$

The achievable performance metric is then defined as:

$$ R_t = (1 - \rho_t) \log_2(1 + \text{SINR}_t) \quad (11) $$

It is important to note that $R_t$ represents the normalized spectral efficiency per subcarrier rather than the absolute data rate. The actual throughput can be obtained by scaling with system bandwidth and symbol duration.

This reformulation ensures consistency between the receive signal model and the interference characterization, while maintaining analytical tractability for reinforcement learning-based optimization.

## 4 CFO Estimator Normalization and Underlying Assumptions

The carrier frequency offset (CFO) is estimated using the phase difference between consecutive OFDM symbols on pilot subcarriers. The original formulation employs the estimator:

$$ \hat{\epsilon}_t = \frac{1}{2\pi T_s} \cdot \frac{1}{|P_t|N} \sum_{k \in P_t} \sum_{n=1}^{N-1} \angle \left( \frac{Y_t[k, n+1]}{Y_t[k, n]} \right) \quad (12) $$

However, this normalization is incorrect. The inner summation spans $N - 1$ terms, and therefore division by $N$ introduces a systematic bias. Specifically, the estimator is scaled by a factor of $(N - 1)/N$, resulting in underestimation of the true CFO.

To obtain an unbiased estimator, the normalization must be corrected as:

$$ \hat{\epsilon}_t = \frac{1}{2\pi T_s} \cdot \frac{1}{|P_t|(N - 1)} \sum_{k \in P_t} \sum_{n=1}^{N-1} \angle \left( \frac{Y_t[k, n+1]}{Y_t[k, n]} \right) \quad (13) $$

In addition to normalization, the validity of this estimator relies on several implicit assumptions that must be explicitly stated.

First, the pilot symbols must remain identical across consecutive OFDM symbols, i.e.,
$$ X_t[k, n + 1] = X_t[k, n], \forall k \in P_t \quad (14) $$
This ensures that the modulation component cancels in the ratio.

Second, the channel is assumed to be quasi-static over adjacent OFDM symbols:
$$ H_t[k, n + 1] \approx H_t[k, n] \quad (15) $$
Under this assumption, the channel terms cancel, isolating the CFO-induced phase rotation.

Third, the estimator assumes moderate-to-high signal-to-noise ratio (SNR), such that noise-induced phase perturbations do not dominate the angle operation.

Under these conditions, the ratio simplifies to:
$$ \frac{Y_t[k, n + 1]}{Y_t[k, n]} \approx e^{j2\pi\epsilon_t T_s} \quad (16) $$
and the CFO can be reliably extracted from the phase.

This corrected formulation ensures that the estimator is unbiased and that all underlying assumptions are explicitly stated, thereby improving both theoretical rigor and practical validity.

## 5 Consistency of CFO Units

The original formulation describes the carrier frequency offset (CFO) $\epsilon_t$ as a normalized quantity, while simultaneously employing the phase model:
$$ e^{j2\pi\epsilon_t n T_s} \quad (17) $$
This introduces a dimensional inconsistency. If $\epsilon_t$ is normalized (dimensionless), the inclusion of the OFDM symbol duration $T_s$ results in incorrect scaling. Conversely, if the phase expression includes $T_s$, then $\epsilon_t$ must represent a physical frequency offset measured in Hertz.

To ensure dimensional consistency, we adopt the interpretation of $\epsilon_t$ as a physical CFO in Hertz. Under this definition, the phase evolution across OFDM symbols is correctly given by:
$$ e^{j2\pi\epsilon_t n T_s} \quad (18) $$
where $T_s$ denotes the OFDM symbol duration.

Accordingly, the CFO estimator is expressed as:
$$ \hat{\epsilon}_t = \frac{1}{2\pi T_s} \cdot \frac{1}{|P_t|(N - 1)} \sum_{k \in P_t} \sum_{n=1}^{N-1} \angle \left( \frac{Y_t[k, n + 1]}{Y_t[k, n]} \right) \quad (19) $$
Under this formulation, the argument of the angle operator corresponds to $2\pi\epsilon_t T_s$, ensuring that the estimator recovers the CFO in Hertz.

Alternatively, a normalized CFO representation may be adopted by defining $\epsilon_t^{(n)} = \epsilon_t T_s$, in which case the phase term becomes $e^{j2\pi\epsilon_t^{(n)} n}$ and the estimator must be modified accordingly. However, both representations must not be mixed within the same model.

This clarification ensures dimensional consistency across the signal model and estimator, and aligns the formulation with standard practices in OFDM synchronization analysis.

## 6 Consistent Sensing Model and State Definition

The original sensing model defines the delay estimation variance as:
$$ \text{Var}(\hat{\tau}_t) \geq \frac{1}{8\pi^2 B^2 \cdot \text{SNR}_{\text{pilot},t}} \quad (20) $$
with
$$ \text{SNR}_{\text{pilot},t} = \frac{\beta_t P_t}{\sigma^2} \quad (21) $$
This formulation is incomplete, as it neglects the effect of pilot density. In the proposed system, pilot resources are distributed across $\rho_t K$ subcarriers, and therefore the effective sensing signal-to-noise ratio must account for both pilot power and pilot allocation.

Under a flat-fading assumption, the effective pilot-domain SNR is more accurately expressed as:
$$ \text{SNR}_{\text{pilot},t} = \frac{\beta_t P_t |H_t|^2}{\rho_t K \sigma^2} \quad (22) $$
This expression reflects the tradeoff between pilot power and pilot density, and ensures consistency with the earlier claim that both $\rho_t$ and $\beta_t$ jointly influence sensing performance.

Furthermore, the use of the Cramér–Rao Lower Bound (CRLB) in the reward function introduces a conceptual inconsistency. The CRLB represents a theoretical performance bound and is not directly measurable during online operation. Reinforcement learning requires the reward to be computed from observable quantities.

To address this issue, we replace the CRLB-based term with a measurable sensing performance metric. For example, a proxy based on the sharpness of the delay correlation function or an empirical delay estimation error can be used. Denoting such a measurable metric as $S_t$, the reward formulation remains consistent with practical implementation.

In addition, the original state definition includes an ambiguous variable $SNR_t$, which is not clearly defined. To eliminate this ambiguity, we explicitly distinguish between communication and sensing quality metrics. The state therefore incorporates the estimated communication SINR and/or pilot-domain SNR, depending on the system objective.

A consistent state representation is thus given by:
$$ s_t = \left( \hat{\epsilon}_{t-1}, \mathcal{E}_{t-1}, \overline{\text{SINR}}_{t-1}, \overline{\text{SNR}}_{\text{pilot},t-1}, |\hat{H}_{t-1}|^2, \rho_{t-1}, \beta_{t-1} \right) \quad (23) $$
This formulation ensures that all state variables are measurable, all performance metrics are physically meaningful, and the sensing model is consistent with the system resource allocation structure.

## 7 Convergence and Stability Considerations

The original formulation includes strong claims regarding the convergence and stability of the Deep Q-Network (DQN) algorithm. However, such claims require careful qualification. In general, DQN employs nonlinear function approximation, bootstrapped updates, and off-policy learning, for which no universal convergence guarantees exist. As a result, theoretical convergence to the optimal action-value function cannot be assured.

Instead, DQN is known to exhibit stable empirical performance under appropriate design choices, including the use of target networks and experience replay. Therefore, in this work, we restrict our claims to empirical convergence observed through simulation rather than theoretical guarantees.

Another important consideration is the boundedness of the reward function. The original formulation models the CFO evolution as a random walk:
$$ \epsilon_{t+1} = \epsilon_t + \nu_t \quad (24) $$
which is an unbounded process. This can lead to unbounded synchronization error and, consequently, unbounded negative reward values. Such behavior violates standard assumptions required for stable reinforcement learning.

To address this issue, we introduce a bounded and mean-reverting CFO model:
$$ \epsilon_{t+1} = \alpha\epsilon_t + \nu_t, \quad |\alpha| < 1 \quad (25) $$
which ensures that the CFO remains within a finite range in steady state.

In addition, the reward function is constructed using bounded and measurable performance metrics, ensuring that:
$$ r_t \in [r_{\min}, r_{\max}] \quad (26) $$
This boundedness improves numerical stability and aligns with standard reinforcement learning assumptions.

With these modifications, the proposed framework satisfies the practical conditions under which DQN is known to perform reliably. While formal convergence guarantees remain intractable, the combination of bounded rewards, finite action space, and stabilized training mechanisms enables robust and consistent learning behavior in practice.