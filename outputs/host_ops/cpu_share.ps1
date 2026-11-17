# Read-only: cores a process uses (its CPU time over S seconds), the host's logical cores and its CPU load.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File outputs/host_ops/cpu_share.ps1 PID S
$p = [int]$args[0]; $s = [int]$args[1]
function cputime($id) {
    $w = Get-CimInstance Win32_Process -Filter "ProcessId=$id"
    if (-not $w) { return $null }
    return ([double]$w.KernelModeTime + [double]$w.UserModeTime) / 1e7
}
$n = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$a = cputime $p; $t0 = Get-Date
Start-Sleep -Seconds $s
$b = cputime $p; $dt = ((Get-Date) - $t0).TotalSeconds
$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$th = (Get-Process -Id $p -ErrorAction SilentlyContinue).Threads.Count
'logical cores {0}; host load {1}%; pid {2}: {3:N2} cores over {4:N0} s, {5} threads' -f $n, $load, $p, (($b - $a) / $dt), $dt, $th
