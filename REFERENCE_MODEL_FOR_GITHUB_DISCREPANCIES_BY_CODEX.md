# Discrepancies Noted by Codex in `REFERENCE_MODEL_FOR_GITHUB.md`

This note records the discrepancies I found while critically reviewing `REFERENCE_MODEL_FOR_GITHUB.md`.

## Findings

### 1. Receive model and later SINR/ICI model are not fully consistent

- Severity: High
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:30`
  - `REFERENCE_MODEL_FOR_GITHUB.md:39`
  - `REFERENCE_MODEL_FOR_GITHUB.md:90`
  - `REFERENCE_MODEL_FOR_GITHUB.md:164`

Equation (3) models CFO as a per-symbol phase rotation on each subcarrier, but Sections 3.1-3.2 later treat residual CFO as creating inter-carrier interference (ICI). In OFDM, CFO that generates ICI normally appears as coupling across subcarriers in the receive model itself. That coupling is absent here. There is also a shift from the channel model `H_t[k,n]` to a scalar `H_t` or `|H_t|^2` later in the document, without explaining the reduction.

### 2. The state and reward use a quantity that is not directly measurable

- Severity: High
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:77`
  - `REFERENCE_MODEL_FOR_GITHUB.md:116`
  - `REFERENCE_MODEL_FOR_GITHUB.md:138`
  - `REFERENCE_MODEL_FOR_GITHUB.md:141`

The document defines the residual error as `\tilde{\epsilon}_t = \epsilon_t - \hat{\epsilon}_t` and then uses `\tilde{\epsilon}_t^2` in both the reward and the MDP state. But the true CFO `\epsilon_t` is not directly known at the receiver, so `\tilde{\epsilon}_t^2` is not directly measurable in the claimed online setting. This conflicts with the statement that the state contains measurable quantities and weakens the Markov-state claim as written.

### 3. CFO units are inconsistent

- Severity: High
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:34`
  - `REFERENCE_MODEL_FOR_GITHUB.md:55`
  - `REFERENCE_MODEL_FOR_GITHUB.md:60`

Line 34 describes `\epsilon_t` as a normalized CFO, but Equations (3), (6), and (7) multiply it by `T_s`. If `\epsilon_t` is normalized, `T_s` should not appear in that phase term. If `T_s` remains, then `\epsilon_t` should instead be interpreted as a frequency offset in Hz rather than a normalized offset.

### 4. The CFO estimator normalization and assumptions are incomplete

- Severity: Medium
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:53`
  - `REFERENCE_MODEL_FOR_GITHUB.md:60`
  - `REFERENCE_MODEL_FOR_GITHUB.md:63`

Equation (7) sums over `n = 1` to `N - 1`, but the normalization term uses `|\mathcal{P_t}|N` instead of `|\mathcal{P_t}|(N - 1)`. Also, the ratio `Y_t[k,n+1] / Y_t[k,n]` only isolates CFO cleanly if pilot symbols repeat across consecutive OFDM symbols and the channel stays approximately constant over those symbols. Those assumptions are not stated in the document.

### 5. The sensing model does not depend on pilot ratio `\rho_t`

- Severity: Medium
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:103`
  - `REFERENCE_MODEL_FOR_GITHUB.md:107`
  - `REFERENCE_MODEL_FOR_GITHUB.md:110`

Equations (13) and (14) make sensing performance depend only on pilot power fraction `\beta_t`. However, the surrounding text claims adaptive pilot allocation jointly affects sensing and communication. As written, pilot ratio `\rho_t` does not influence the sensing bound, so the “joint” dependence is incomplete.

### 6. The “achievable rate per frame” expression is dimensionally loose

- Severity: Medium
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:81`
  - `REFERENCE_MODEL_FOR_GITHUB.md:90`
  - `REFERENCE_MODEL_FOR_GITHUB.md:94`

Equation (12) is described as an achievable rate per frame, but it is more like a normalized spectral-efficiency expression. It omits factors such as `K`, `N`, symbol duration, or bandwidth that would be needed for an actual per-frame or per-second rate. Also, Equation (10) scales ICI with total transmit power `P_t` but not with channel gain, while the desired signal term in Equation (11) does include `|H_t|^2`. That mismatch is physically questionable.

### 7. The convergence discussion is overstated

- Severity: Medium
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:190`
  - `REFERENCE_MODEL_FOR_GITHUB.md:218`

Section 9 states convergence and stability properties too strongly. Standard DQN with nonlinear function approximation does not have a general convergence guarantee in the form claimed there. The reward is also not obviously bounded if CFO follows an unbounded random walk, so the theoretical justification is weaker than the text suggests.

### 8. There are notation and LaTeX consistency issues in the GitHub version

- Severity: Low
- Relevant lines:
  - `REFERENCE_MODEL_FOR_GITHUB.md:12`
  - `REFERENCE_MODEL_FOR_GITHUB.md:17`
  - `REFERENCE_MODEL_FOR_GITHUB.md:22`
  - `REFERENCE_MODEL_FOR_GITHUB.md:141`

Equation (1) uses `|\mathcal{P}|`, while Equations (2) and (7) use `\mathcal{P_t}`. If the pilot set is frame-dependent, it should be written consistently as `\mathcal{P}_t`. Likewise, `\mathcal{D_t}` should be `\mathcal{D}_t`. The text around line 141 also contains odd spaced subscripts such as `\hat{\epsilon} _ t`, `\rho _ {t-1}`, and `\beta _ {t-1}`, which are poor LaTeX style and may render badly.

## Summary

The main issues are not just cosmetic. The strongest discrepancies are:

- mismatch between the receive model and the later ICI-based SINR model,
- use of non-measurable residual CFO in the state and reward,
- inconsistent CFO units,
- overstated convergence claims.

The notation issues are secondary, but they should still be cleaned up because they make the GitHub version look less rigorous.
