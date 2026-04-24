param(
    [ValidateSet("smoke", "focused_e10")]
    [string]$Profile = "smoke",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

$argsList = @()

switch ($Profile) {
    "smoke" {
        $argsList = @(
            "--frames", "8",
            "--eval-episodes", "1",
            "--seeds", "7",
            "--weights", "1.0:0.2:0.4",
            "--regime-switch-prob", "0.05",
            "--noise-variance", "0.05",
            "--cfo-high-drift-std", "100",
            "--cfo-high-drift-alpha", "0.7",
            "--workers", "1",
            "--output-dir", "artifacts/ec2_oracle_sweep_smoke"
        )
    }
    "focused_e10" {
        $workerCount = if ($env:ORACLE_SWEEP_WORKERS) { $env:ORACLE_SWEEP_WORKERS } else { "4" }
        $argsList = @(
            "--frames", "64",
            "--eval-episodes", "10",
            "--seeds", "7,17,27,37,47",
            "--weights", "1.0:0.2:0.4,1.0:0.25:0.5,1.0:0.4:0.5,1.0:0.5:0.4",
            "--regime-switch-prob", "0.02,0.05",
            "--noise-variance", "0.05,0.1",
            "--cfo-high-drift-std", "300,600,1000",
            "--cfo-high-drift-alpha", "0.7",
            "--workers", $workerCount,
            "--output-dir", "artifacts/phase2b_oracle_sweep_focused_e10"
        )
    }
}

if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

& $pythonBin -m ofdm_isac_rl.oracle_sweep @argsList
