# Read-only: N readings, each over S seconds, of the cores a process uses, the host's CPU % and the card's util and power.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File outputs/host_ops/cpu_share_series.ps1 PID N S
$p = [int]$args[0]; $n = [int]$args[1]; $s = [int]$args[2]
function cputime($id) {
    $w = Get-CimInstance Win32_Process -Filter "ProcessId=$id"
    if (-not $w) { return $null }
    return ([double]$w.KernelModeTime + [double]$w.UserModeTime) / 1e7
}
$ctr = '\Processor(_Total)\% Processor Time'
for ($i = 0; $i -lt $n; $i++) {
    $a = cputime $p; $t0 = Get-Date
    $host_pct = (Get-Counter $ctr -SampleInterval $s -MaxSamples 1).CounterSamples[0].CookedValue
    $b = cputime $p; $dt = ((Get-Date) - $t0).TotalSeconds
    if ($a -eq $null -or $b -eq $null) { "pid $p gone"; break }
    $g = nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm --format=csv,noheader
    '{0} pid {1}: {2:N2} cores | host {3:N0}% | card {4}' -f (Get-Date -Format 'HH:mm:ss'), $p, (($b - $a) / $dt), $host_pct, $g
}
