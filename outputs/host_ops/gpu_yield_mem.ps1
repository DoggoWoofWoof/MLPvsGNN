# Read-only: per-process GPU memory (dedicated and shared) for the pids given, then the card's util, power and memory.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File outputs/host_ops/gpu_yield_mem.ps1 13956 44576
$pids = $args
$c = Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
foreach ($s in $c.CounterSamples) {
    foreach ($p in $pids) {
        if ($s.InstanceName.StartsWith("pid_$($p)_") -and $s.CookedValue -gt 50MB) {
            $kind = if ($s.Path.EndsWith('dedicated usage')) { 'dedicated' } else { 'shared' }
            '{0} {1} {2:N1} GB' -f $p, $kind, ($s.CookedValue / 1GB)
        }
    }
}
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm --format=csv,noheader
