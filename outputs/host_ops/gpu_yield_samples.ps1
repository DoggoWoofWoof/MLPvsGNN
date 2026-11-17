# Read-only: N samples of the card's util, power, memory and clocks, S seconds apart.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File outputs/host_ops/gpu_yield_samples.ps1 12 3
$n = [int]$args[0]; $s = [int]$args[1]
for ($i = 0; $i -lt $n; $i++) {
    $t = Get-Date -Format 'HH:mm:ss'
    "$t " + (nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used,clocks.sm --format=csv,noheader)
    Start-Sleep -Seconds $s
}
