# One-time WSL configuration for rx jobs that use --shell wsl. Run it yourself, as the host
# account's owner, from the project root on your machine:
#
#   python tools/rx/rx.py push
#   ssh gpu "powershell -NoProfile -ExecutionPolicy Bypass -File rx\projects\mpr\ws\tools\rx\wsl_setup.ps1"
#
# It changes system and security settings, which is why rx never runs it for you:
#   - creates the Linux user (default: your Windows user name, lower-cased) with passwordless sudo,
#     checked with visudo before it is installed;
#   - writes /etc/wsl.conf: systemd, that user as the default, no Windows PATH, metadata on /mnt/c
#     (an existing file is kept as /etc/wsl.conf.rx-backup);
#   - writes %USERPROFILE%\.wslconfig: the VM limits below (an existing, different file is kept
#     as .wslconfig.rx-backup-<time>);
#   - runs `wsl --shutdown`, which stops every running WSL distro of THIS Windows account only,
#     then starts the distro again and checks each setting.
# Safe to run again: every step checks before it writes.
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$User = $env:USERNAME.ToLower(),
    [string]$Memory = '96GB',
    [int]$Processors = 32,
    [string]$Swap = '32GB'
)
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
if ($User -notmatch '^[a-z_][a-z0-9_-]{0,31}$') { throw "'$User' is not a valid Linux user name; pass -User NAME" }

function Invoke-Wsl([string[]]$WslArgs) {
    & wsl.exe @WslArgs
    if ($LASTEXITCODE -ne 0) { throw "wsl.exe $($WslArgs -join ' ') exited with $LASTEXITCODE" }
}

function Write-LinuxScript([string]$Name, [string]$Text) {
    $win = Join-Path (Join-Path $env:USERPROFILE 'rx') $Name
    [IO.File]::WriteAllText($win, ($Text -replace "`r`n", "`n"))
    return @{ win = $win; lin = '/mnt/' + $win.Substring(0, 1).ToLower() + ($win.Substring(2) -replace '\\', '/') }
}

$names = (& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
if ($names -notcontains $Distro) { throw "no WSL distro named $Distro (have: $($names -join ', '))" }

# 1. inside the distro, as root: the user, sudo, /etc/wsl.conf
$root = @'
#!/bin/bash
set -euo pipefail
U="$1"
if ! id -u "$U" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "$U"
  echo "created user $U"
fi
tmp=$(mktemp)
printf '%s ALL=(ALL) NOPASSWD:ALL\n' "$U" > "$tmp"
visudo -cf "$tmp" >/dev/null
install -m 0440 -o root -g root "$tmp" "/etc/sudoers.d/90-rx-$U"
rm -f "$tmp"
if [ -f /etc/wsl.conf ] && [ ! -f /etc/wsl.conf.rx-backup ]; then cp -a /etc/wsl.conf /etc/wsl.conf.rx-backup; fi
cat > /etc/wsl.conf <<EOF
[boot]
systemd=true

[user]
default=$U

[interop]
enabled=true
appendWindowsPath=false

[automount]
enabled=true
options="metadata,umask=22,fmask=11"
EOF
echo "root setup done: $(id "$U")"
'@
$s = Write-LinuxScript 'wsl_root_setup.sh' $root
Invoke-Wsl @('-d', $Distro, '-u', 'root', '--', 'bash', $s.lin, $User)

# 2. the VM limits for this Windows account
$cfg = Join-Path $env:USERPROFILE '.wslconfig'
$want = @"
# rx: WSL2 VM limits for this account (read when the VM starts; apply with wsl --shutdown)
[wsl2]
memory=$Memory
processors=$Processors
swap=$Swap

[experimental]
autoMemoryReclaim=dropCache
sparseVhd=true
"@ -replace "`r`n", "`n"
if (Test-Path $cfg) {
    $have = [IO.File]::ReadAllText($cfg) -replace "`r`n", "`n"
    if ($have.Trim() -ne $want.Trim()) {
        $bak = "$cfg.rx-backup-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
        Copy-Item $cfg $bak
        "existing .wslconfig kept as $bak; it read:"
        $have
        [IO.File]::WriteAllText($cfg, $want)
        "wrote $cfg"
    } else { "$cfg already current" }
} else {
    [IO.File]::WriteAllText($cfg, $want)
    "wrote $cfg"
}

# 3. restart this account's WSL VM so both files take effect, then check every setting
Invoke-Wsl @('--shutdown')
Start-Sleep -Seconds 10
$verify = @'
#!/bin/bash
U="$1"; MEM="$2"; CPUS="$3"; SWAP="$4"
bad=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1: $2"; else echo "FAIL $1: $2 (want $3)"; bad=1; fi; }
check user "$(whoami)" "$U"
check pid1 "$(ps -p 1 -o comm=)" systemd
check sudo "$(sudo -n true 2>/dev/null && echo yes || echo no)" yes
check cpus "$(nproc)" "$CPUS"
mem_gb=$(awk '/MemTotal/{printf "%d", ($2 / 1048576) + 0.5}' /proc/meminfo)
swap_gb=$(awk '/SwapTotal/{printf "%d", ($2 / 1048576) + 0.5}' /proc/meminfo)
check memory_near_limit "$(( mem_gb >= MEM - 4 && mem_gb <= MEM ))" 1     # the kernel keeps ~2 GB
check swap_gb "$swap_gb" "$SWAP"
check mnt_c_metadata "$(grep -c ' /mnt/c .*metadata' /proc/mounts)" 1
check windows_path "$(echo "$PATH" | tr ':' '\n' | grep -c '^/mnt/')" 0
echo "info memory ${mem_gb} GB, systemd state: $(systemctl is-system-running 2>/dev/null)"
echo "info gpu: $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)"
if [ "$bad" = 0 ]; then echo "WSL SETUP OK"; else echo "WSL SETUP INCOMPLETE"; exit 1; fi
'@
$v = Write-LinuxScript 'wsl_verify.sh' $verify
& wsl.exe -d $Distro -- bash $v.lin $User ([int]($Memory -replace 'GB', '')) $Processors ([int]($Swap -replace 'GB', ''))
$rc = $LASTEXITCODE
Remove-Item -Force $s.win, $v.win
exit $rc
