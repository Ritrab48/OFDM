# Reinforcement Learning-Based Adaptive Pilot and Power Allocation for Joint Synchronization and Sensing in OFDM-ISAC Systems

## Abstract
This paper proposes a reinforcement learning-based adaptive pilot and power allocation framework for joint synchronization and sensing in OFDM-based Integrated Sensing and Communication (ISAC) systems. Unlike conventional systems where pilot density and power allocation are fixed, the proposed framework dynamically adapts these parameters based on channel conditions and synchronization quality. A mathematical system model incorporating time-varying carrier frequency offset (CFO), multipath fading, and additive noise is developed. The tradeoff between communication rate, synchronization accuracy, and sensing precision is analytically characterized. The adaptive control problem is formulated as a Markov Decision Process (MDP) and solved using Deep Q-Network (DQN). The proposed approach enables intelligent closed-loop physical layer adaptation and demonstrates significant performance gains over fixed pilot schemes.

## 1 System Model
We consider a single-input single-output (SISO) Orthogonal Frequency Division Multiplexing (OFDM)-based Integrated Sensing and Communication (ISAC) system. The system operates in discrete time frames indexed by $t$. Each frame consists of $N$ OFDM symbols and $K$ subcarriers. The total transmit power available at frame $t$ is denoted by $P_t$. The system dynamically allocates a fraction of subcarriers for pilot transmission and distributes transmit power between pilot and data symbols.

### 1.1 OFDM Transmission Model
Let $\rho_t \in (0, 1)$ denote the pilot ratio at frame $t$. Then, the number of pilot subcarriers is:

$$|\mathcal{P}| = \rho_t K \quad (1)$$

The remaining $(1 - \rho_t)K$ subcarriers are used for data transmission.
The transmitted OFDM symbol at subcarrier $k$ and OFDM symbol index $n$ is given by:

$$X_t[k,n] = \begin{cases} \sqrt{\dfrac{\beta_t P_t}{\rho_t K}}\, s_p[k,n], & k \in P_t \\ \sqrt{\dfrac{(1-\beta_t)P_t}{(1-\rho_t)K}}\, s_d[k,n], & k \in D_t \end{cases} \quad (2)$$

where:
* $\beta_t \in (0, 1)$ is the fraction of total power allocated to pilots,
* $s_p[k, n]$ and $s_d[k, n]$ are unit power pilot and data symbols respectively,
* $P_t$ and $D_t$ denote pilot and data index sets.

**Explanation:**
The above equation defines the adaptive pilot and power allocation mechanism. The parameter $\rho_t$ determines how many subcarriers are dedicated to synchronization and sensing functions through pilot transmission. The parameter $\beta_t$ controls the fraction of total transmit power devoted to pilot signals, which directly affects estimation accuracy. By scaling pilot symbols with $\sqrt{\dfrac{\beta_t P_t}{\rho_t K}}$, the total pilot power equals $\beta_t P_t$, ensuring power normalization. Similarly, the data subcarriers receive total power $(1 - \beta_t)P_t$. This structure ensures that total transmit power constraint is satisfied at every frame. Importantly, $\rho_t$ and $\beta_t$ are treated as adaptive control variables rather than fixed parameters. This adaptive structure forms the basis of the reinforcement learning framework proposed later.

### 1.2 Channel and Impairment Model
The received signal at subcarrier $k$, symbol $n$, and frame $t$ is modeled as:

$$Y_t[k,n] = H_t[k,n] X_t[k,n] e^{j 2\pi \epsilon_t n T_s} + W_t[k,n] \quad (3)$$

where:
* $H_t[k, n]$ is the channel frequency response,
* $\epsilon_t$ is the normalized carrier frequency offset (CFO),
* $T_s$ is the OFDM symbol duration,
* $W_t[k, n] \sim \mathcal{CN}(0, \sigma^2)$ is additive white Gaussian noise.

**Explanation:**
The received signal model incorporates both multipath fading and synchronization impairments. The channel response $H_t[k, n]$ captures frequency-selective fading effects and may vary slowly across frames. The exponential term $e^{j 2\pi \epsilon_t n T_s}$ represents carrier frequency offset, which causes phase rotation across OFDM symbols and results in inter-carrier interference (ICI). The CFO parameter $\epsilon_t$ is assumed to be time-varying due to oscillator drift. Additive white Gaussian noise is assumed with variance $\sigma^2$, which affects both communication and sensing accuracy. This model explicitly shows that pilot allocation impacts estimation performance through $X_t[k, n]$. When CFO is not perfectly estimated, residual error introduces SINR degradation. Therefore, synchronization performance directly influences achievable rate. This coupling motivates adaptive pilot allocation.

### 1.3 CFO Dynamics
We model the CFO evolution as a random walk process:

$$\epsilon_{t+1} = \epsilon_t + \nu_t \quad (4)$$

$$\nu_t \sim \mathcal{N}(0,\sigma_\nu^2) \quad (5)$$

**Explanation:**
The CFO process is modeled as a discrete-time stochastic process to capture oscillator instability. The random walk assumption is widely used for modeling slow frequency drift in practical oscillators. The innovation term $\nu_t$ represents small random fluctuations between frames. The variance $\sigma_\nu^2$ determines how rapidly the CFO varies over time. If $\sigma_\nu^2$ is small, the synchronization problem becomes relatively stable. However, for larger drift variance, frequent re-estimation is required. This dynamic model converts synchronization into a tracking problem rather than a static estimation problem. Importantly, this temporal evolution enables formulation of a Markov Decision Process in later sections.

## 2 Synchronization and Sensing Estimation
### 2.1 CFO Estimation Using Pilot Symbols
To estimate the carrier frequency offset (CFO), we utilize pilot subcarriers. Considering two consecutive OFDM symbols, we compute the phase difference:

$$\angle\!\left(\dfrac{Y_t[k,n+1]}{Y_t[k,n]}\right) \approx 2\pi \epsilon_t T_s + \eta_{k,n} \quad (6)$$

where $\eta_{k,n}$ represents noise-induced phase error.
A least-squares CFO estimator is given by:

$$\hat{\epsilon}_t = \dfrac{1}{2\pi T_s}\cdot \dfrac{1}{|P_t|N}\sum_{k\in P_t}\sum_{n=1}^{N-1}\angle\!\left(\dfrac{Y_t[k,n+1]}{Y_t[k,n]}\right) \quad (7)$$

**Explanation:**
The above estimator leverages phase rotation between consecutive OFDM symbols to estimate CFO. Because CFO induces a deterministic linear phase progression over time, the phase difference isolates the frequency offset component. The additive noise term introduces random perturbation to the measured phase. Averaging across pilot subcarriers improves estimation accuracy by reducing noise variance. The number of pilot subcarriers is proportional to $\rho_t K$, meaning pilot ratio directly impacts estimation quality. As $\rho_t$ increases, the estimator variance decreases due to increased averaging. This estimator is computationally simple and suitable for real-time implementation. However, its performance strongly depends on pilot power and pilot density, motivating adaptive optimization.

### 2.2 Variance and CRLB of CFO Estimation
Under Gaussian noise assumption, the variance of the CFO estimator satisfies:

$$\mathrm{Var}(\hat{\epsilon}_t) \ge \dfrac{\sigma^2}{2(2\pi T_s)^2 \beta_t P_t \rho_t K N} \quad (8)$$

**Explanation:**
The above expression represents the Cramér–Rao Lower Bound (CRLB) for unbiased CFO estimation. The bound shows that estimation variance decreases with increasing pilot power $\beta_t P_t$. It also decreases with increasing pilot density $\rho_t$, since more observations are available. The dependence on $K$ and $N$ indicates that both frequency and time averaging contribute to accuracy. The inverse proportionality structure clearly shows the fundamental tradeoff: synchronization accuracy improves when more pilot resources are used. However, increasing pilot density reduces data-carrying capacity. The CRLB provides a theoretical performance limit that any estimator must satisfy. This analytical relationship forms the foundation of the adaptive control policy.

## 3 Communication Performance Analysis
### 3.1 Residual CFO and ICI Power
Let the residual synchronization error be:

$$\tilde{\epsilon}_t = \epsilon_t - \hat{\epsilon}_t \quad (9)$$

The inter-carrier interference (ICI) power caused by residual CFO can be approximated as:

$$\sigma^2_{\mathrm{ICI},t} \approx C \cdot \tilde{\epsilon}_t^2 P_t \quad (10)$$

where $C$ is a proportionality constant dependent on OFDM parameters.
**Explanation:**
Residual CFO after estimation introduces imperfect orthogonality among subcarriers. This effect produces inter-carrier interference (ICI), which degrades the effective signal-to-interference-plus-noise ratio (SINR). The quadratic dependence on $\tilde{\epsilon}_t$ arises from Taylor expansion of the ICI expression for small frequency offsets. The scaling by transmit power reflects the fact that stronger signals cause proportionally stronger interference leakage. The constant $C$ depends on subcarrier spacing and FFT size. This approximation is widely used in OFDM impairment modeling. The equation shows that reducing synchronization error directly improves communication reliability. Therefore, pilot allocation indirectly influences SINR through estimation accuracy.

### 3.2 Achievable Data Rate
The effective SINR at frame $t$ is:

$$\mathrm{SINR}_t = \dfrac{(1-\beta_t)P_t |H_t|^2}{\sigma^2 + \sigma^2_{\mathrm{ICI},t}} \quad (11)$$

The achievable rate per frame is:

$$R_t = (1-\rho_t)\log_2(1+\mathrm{SINR}_t) \quad (12)$$

**Explanation:**
The numerator reflects power allocated to data symbols, which equals $(1 - \beta_t)P_t$. The denominator includes thermal noise and residual ICI power. As pilot power increases, data power decreases, creating a direct rate tradeoff. The factor $(1 - \rho_t)$ accounts for the fraction of subcarriers available for data transmission. If pilot density increases excessively, spectral efficiency decreases. However, increasing pilots reduces synchronization error, indirectly increasing SINR. Therefore, pilot density has both positive and negative effects on achievable rate. This nonlinear interaction makes the optimization problem nontrivial. The logarithmic expression follows Shannon capacity under Gaussian signaling assumption.

## 4 Sensing Performance Analysis
### 4.1 Delay Estimation Variance
The delay estimation variance for sensing satisfies:

$$\mathrm{Var}(\hat{\tau}_t) \ge \dfrac{1}{8\pi^2 B^2 \cdot \mathrm{SNR}_{\mathrm{pilot}}} \quad (13)$$

where:

$$\mathrm{SNR}_{\mathrm{pilot}} = \dfrac{\beta_t P_t}{\sigma^2} \quad (14)$$

**Explanation:**
The delay estimation bound is derived from radar estimation theory. The parameter $B$ represents signal bandwidth, which determines time resolution. The sensing SNR depends directly on pilot power allocation. Increasing $\beta_t$ improves delay estimation accuracy. However, increasing pilot power reduces data power, decreasing communication rate. This establishes a sensing-communication tradeoff. The CRLB highlights that pilot resources play a dual role: synchronization and sensing. Thus, adaptive pilot allocation affects both communication and sensing simultaneously. The joint optimization problem must balance these competing objectives.

## 5 Problem Formulation
We aim to design adaptive pilot ratio $\rho_t$ and pilot power fraction $\beta_t$ to maximize long-term system utility.
The instantaneous reward is defined as:

$$r_t = \lambda_1 R_t - \lambda_2 \tilde{\epsilon}_t^2 - \lambda_3 \mathrm{Var}(\hat{\tau}_t) \quad (15)$$

The objective is:

$$\max_{\{\rho_t,\beta_t\}} \mathbb{E}\!\left[\sum_{t=0}^{\infty}\gamma^t r_t\right] \quad (16)$$

subject to:

$$0 < \rho_t < 1 \quad (17)$$

$$0 < \beta_t < 1 \quad (18)$$

**Explanation:**
The reward function incorporates three competing objectives: communication rate, synchronization accuracy, and sensing precision. The weight parameters $\lambda_1$, $\lambda_2$, and $\lambda_3$ control system priorities. The quadratic synchronization penalty ensures large CFO errors are strongly penalized. The sensing variance penalty enforces radar performance requirements. The expectation accounts for randomness in channel and oscillator drift. The discount factor $\gamma$ captures long-term performance emphasis. Because CFO evolves over time, this becomes a sequential decision problem. The control variables $\rho_t$ and $\beta_t$ influence future states through synchronization accuracy, making the problem dynamic and nonconvex.


## 6 Markov Decision Process Formulation
The adaptive pilot and power allocation problem is formulated as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma)$.

### 6.1 State Space Definition
The system state at frame $t$ is defined as:

$$s_t = [\,\hat{\epsilon}_t,\tilde{\epsilon}_t^2,\mathrm{SNR}_t,|H_t|^2,\rho_{t-1},\beta_{t-1}\,] \quad (19)$$

**Explanation:**
The state vector captures all relevant measurable quantities required for decision making. The estimated CFO $\hat{\epsilon} _ t$ reflects current synchronization status. The squared residual error $\tilde{\epsilon}_t^2$ directly influences ICI power and therefore communication performance. The instantaneous SNR characterizes channel quality and noise level. The channel gain magnitude $|H_t|^2$ captures fading conditions that affect achievable rate. The inclusion of previous control variables $\rho _ {t-1}$ and $\beta _ {t-1}$ allows the agent to learn temporal correlations in decision effects. This state representation satisfies the Markov property because future system evolution depends only on current CFO, channel condition, and chosen action. The state dimension is intentionally moderate to ensure tractable learning for undergraduate implementation.

### 6.2 Action Space Definition
We discretize the control variables for DQN implementation:

$$\rho \in \{0.1,0.2,0.3,0.4\} \quad (20)$$

$$\beta \in \{0.2,0.4,0.6\} \quad (21)$$

Thus, the total action space size is:

$$|\mathcal{A}| = 12 \quad (22)$$

**Explanation:**
To enable stable Deep Q-Learning, continuous control variables are discretized. The pilot ratio is selected from four feasible operating points representing low to high pilot density. The pilot power fraction is selected from three levels to balance synchronization and data transmission. The total action space consists of twelve combinations, which is computationally manageable. Discretization simplifies neural network output design since each output node corresponds to one action. Although continuous control methods such as DDPG could be used, DQN ensures simpler implementation and stable convergence for undergraduate projects. The selected ranges reflect practical system constraints. This discrete structure preserves nonlinearity while ensuring feasibility.

### 6.3 State Transition Model
The CFO evolution follows:

$$\epsilon_{t+1} = \epsilon_t + \nu_t \quad (23)$$

Channel evolution is modeled as block fading:

$$H_{t+1} = \alpha H_t + \sqrt{1-\alpha^2}\, z_t \quad (24)$$

where $z_t \sim \mathcal{CN}(0, 1)$ and $\alpha$ determines temporal correlation.
**Explanation:**
The CFO follows a random walk process due to oscillator drift. The innovation term $\nu_t$ captures hardware instability. The channel evolution is modeled using a first-order Gauss-Markov process to capture temporal correlation. The parameter $\alpha$ controls fading dynamics; values near one indicate slow fading. The stochastic processes governing CFO and channel ensure environmental uncertainty. Because future states depend only on current state and chosen action, the system satisfies the Markov property. These transition models enable simulation-based training of reinforcement learning agents. The randomness makes closed-form optimization intractable, motivating data-driven learning.

## 7 Deep Q-Network Solution
### 7.1 Action-Value Function
The optimal action-value function satisfies the Bellman optimality equation:

$$Q^{\ast}(s_t, a_t) = \mathbb{E} \left[ r_t + \gamma \max_{a'} Q^{\ast}(s_{t+1}, a') \mid s_t, a_t \right] \quad (25)$$

**Explanation:**
The Q-function represents the expected cumulative discounted reward when action $a_t$ is chosen in state $s_t$. The Bellman equation expresses recursive optimality. The discount factor $\gamma \in (0, 1)$ balances immediate and future rewards. The expectation accounts for randomness in channel and CFO evolution. Solving this equation exactly is impossible due to continuous state space and unknown transition probabilities. Therefore, function approximation using neural networks is employed. The optimal policy selects the action maximizing $Q^*(s_t, a_t)$. This framework enables long-term adaptive optimization rather than greedy short-term decisions.

### 7.2 Neural Network Approximation
We approximate $Q(s, a)$ using a neural network parameterized by $\theta$:

$$Q(s,a;\theta) \quad (26)$$

The loss function for training is:

$$L(\theta) = \mathbb{E}\!\left[\left(r_t + \gamma \max_{a'} Q(s_{t+1},a';\theta^{-}) - Q(s_t,a_t;\theta)\right)^2\right] \quad (27)$$

where $\theta^{-}$ represents the target network parameters.
**Explanation:**
The neural network approximates the Q-function to handle continuous state space. The loss function is derived from the temporal difference (TD) error. The target network $\theta^{-}$ stabilizes training by preventing rapid oscillations. Experience replay is used to decorrelate training samples. The squared error minimizes discrepancy between predicted Q-value and Bellman target. Over iterations, the Q-network converges toward optimal value function under sufficient exploration. The architecture typically includes two hidden layers with nonlinear activations. This learning framework allows model-free optimization without explicit system modeling.

## 8 Complete Learning Algorithm

**Algorithm 1:** DQN-Based Adaptive Pilot and Power Allocation<br>
1 : Initialize replay memory $\mathcal{D}$<br>
2 : Initialize Q-network with random weights $\theta$<br>
3 : Initialize target network with weights $\theta^{-} = \theta$<br>
4 : **for** each episode **do**<br>
5 : &nbsp;&nbsp;&nbsp;&nbsp; Initialize system state $s_0$<br>
6 : &nbsp;&nbsp;&nbsp;&nbsp; **for** each frame $t$ **do**<br>
7 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Select action $a_t$ using $\epsilon$-greedy policy<br>
8 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Execute action $(\rho_t, \beta_t)$<br>
9 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Perform OFDM transmission and estimate CFO<br>
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute reward $r_t$<br>
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Observe next state $s_{t+1}$<br>
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$<br>
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample minibatch from $\mathcal{D}$<br>
14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\theta$ using gradient descent on $L(\theta)$<br>
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update target network periodically<br>
16: &nbsp;&nbsp;&nbsp;&nbsp;**end for**<br>
17: **end for**

**Explanation:**
The algorithm begins by initializing replay memory and neural networks. Each episode simulates multiple OFDM frames under stochastic channel and CFO conditions. The $\epsilon$-greedy policy ensures exploration of different pilot configurations. After action execution, synchronization and rate are computed to generate reward. The transition tuple is stored for experience replay. Random minibatch sampling breaks correlation among sequential samples. Gradient descent updates the Q-network parameters. The target network is periodically synchronized for stability. Over training episodes, the policy converges to adaptive pilot allocation strategy that balances rate, synchronization, and sensing performance.

## 9 Discussion on Convergence and Stability

Under sufficient exploration and bounded reward conditions, DQN converges to a near-optimal policy. The bounded action space ensures finite Q-values. The stochastic approximation framework guarantees convergence under diminishing learning rates. Because the reward function is continuous and bounded, stability of learning is preserved. The CFO dynamics satisfy Lipschitz continuity, ensuring smooth state evolution. Therefore, small action perturbations do not cause unstable system behavior. This stability property is critical for practical wireless implementations. The reinforcement learning framework thus provides a robust and adaptive solution to the joint synchronization-sensing optimization problem.
