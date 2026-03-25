For the residual-CFO/SINR issue, the clean fix is this:

  $$
  \mathcal E_t \approx (2\pi T_s)^2 p_{\text{pilot}} |H_t|^2 \tilde{\epsilon}_t^2 + 2\sigma^2
  $$

  So instead of using the hidden true residual CFO \tilde{\epsilon}_t = \epsilon_t - \hat{\epsilon}_t, estimate its power from the measurable mismatch
  \mathcal E_t:

  $$
  \widehat{\tilde{\epsilon}_t^2}

  \max\left(
  0,;
  \frac{\mathcal E_t - 2\sigma^2}
  {(2\pi T_s)^2, p_{\text{pilot}}, |\hat H_t|^2}
  \right)
  $$

  with p_pilot = \beta_t P / |\mathcal P_t| in this implementation. Then use that in Equation (8)/(9):

  $$
  \widehat{\sigma^2_{\mathrm{ICI},t}}

  C \cdot \widehat{\tilde{\epsilon}_t^2} \cdot (1-\beta_t)P |\hat H_t|^2
  $$

  $$
  \overline{\mathrm{SINR}}_t

  \frac{(1-\beta_t)P |\hat H_t|^2}
  {\sigma^2 + \widehat{\sigma^2_{\mathrm{ICI},t}}}
  $$

  That is now what the environment does. The estimator is implemented at ofdm_isac_rl/environment.py:297, and the state now uses this observable sinr
  estimate at ofdm_isac_rl/environment.py:107. I kept simulator-truth diagnostics as sinr_true and ici_power_true in ofdm_isac_rl/environment.py:116 so we
  can compare estimated vs true internally.

  This is physically reasonable within the paper’s scope, but it is an approximation, not an identity. It relies on the same assumptions already used around
  Equation (6)/(7): repeated pilots, quasi-static channel across adjacent symbols, and moderate SNR. That makes it a good proof-of-concept fix.

  One important paper-writing point: Equation (9) can stay as the analytical model, but the state in Equation (14) should be interpreted as an observable
  estimate \overline{\mathrm{SINR}}_t, not the hidden exact \mathrm{SINR}_t. Your notation already helps here because Eq. (14) uses the overline.

  Verification passed:

  - python3 -m unittest discover -s tests
  - python3 -m ofdm_isac_rl.train --episodes 2 --frames 8 --eval-episodes 1 --output-dir artifacts/sinr_observable_smoke

  Also, because the environment changed, the old sweep results are now stale. If you want, I’ll rerun the sweep next so the plots and paper pack match the
  corrected SINR model.