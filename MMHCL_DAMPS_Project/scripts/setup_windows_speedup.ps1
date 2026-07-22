#Requires -Version 5.1
<#
.SYNOPSIS
  Windows host setup for PACER / NRDMC-lite training speedups.

.DESCRIPTION
  Implements the Windows-side checklist from
  PACER_NRDMC_lite_training_speedup_guide_EN:

    Step 3 / Sec E.1  -- Windows Defender exclusions
    Step 4 / Sec E.2  -- Ultimate Performance power plan + disable core parking
    Step 16-18 / E.3  -- Ensure CUDA_LAUNCH_BLOCKING is unset (default async)
    Sec C             -- OMP / MKL / PACER thread caps for multi-subprocess grids
    Sec E.4           -- Optional Nsight Systems profile command template

  Run elevated (Administrator) for Defender + powercfg changes.
  Thread / CUDA env vars apply to the current PowerShell session only unless
  -PersistEnv is passed (User-level setx).

.PARAMETER RepoRoot
  Absolute path to the DAMPS_upgrade_for_MMHCL_randoms_Amazon_Clothing repo.
  Defaults to two levels above this script (.../MMHCL_DAMPS_Project/scripts).

.PARAMETER NumThreads
  Per-process CPU thread cap (OMP/MKL/PACER_NUM_THREADS). Default 4.

.PARAMETER PersistEnv
  Also write User-level environment variables via setx.

.PARAMETER SkipDefender
  Skip Add-MpPreference exclusions (useful without Admin / Defender).

.PARAMETER SkipPower
  Skip power-plan / core-parking changes.

.EXAMPLE
  # Elevated PowerShell:
  .\scripts\setup_windows_speedup.ps1

.EXAMPLE
  .\scripts\setup_windows_speedup.ps1 -NumThreads 4 -PersistEnv
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [int]$NumThreads = 4,
    [switch]$PersistEnv,
    [switch]$SkipDefender,
    [switch]$SkipPower
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Set-SessionEnv {
    param([string]$Name, [string]$Value)
    Set-Item -Path "Env:$Name" -Value $Value
    Write-Host "  session  $Name=$Value"
    if ($PersistEnv) {
        # setx truncates at 1024 chars; our values are tiny.
        & setx.exe $Name $Value | Out-Null
        Write-Host "  persisted (User) $Name=$Value"
    }
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
if (-not $RepoRoot) {
    $RepoRoot = (
        Resolve-Path (Join-Path $PSScriptRoot "..\..")
    ).Path
}
$RepoRoot = (Resolve-Path $RepoRoot).Path
$WandbCache = Join-Path $env:USERPROFILE ".cache\wandb"
$InductorCache = Join-Path $env:LOCALAPPDATA "torch_inductor_cache"

Write-Host "=== PACER Windows speedup setup ==="
Write-Host "RepoRoot       : $RepoRoot"
Write-Host "IsAdmin        : $(Test-IsAdmin)"
Write-Host "NumThreads     : $NumThreads"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 3 / Sec E.1 -- Defender exclusions
# ---------------------------------------------------------------------------
if (-not $SkipDefender) {
    Write-Host "[Step 3 / E.1] Windows Defender exclusions"
    if (-not (Test-IsAdmin)) {
        Write-Warning (
            "Not elevated - skipping Defender exclusions. " +
            "Re-run as Administrator, or pass -SkipDefender."
        )
    } else {
        $paths = @($RepoRoot, $WandbCache, $InductorCache)
        foreach ($p in $paths) {
            if (-not (Test-Path -LiteralPath $p)) {
                New-Item -ItemType Directory -Path $p -Force | Out-Null
            }
            try {
                Add-MpPreference -ExclusionPath $p -ErrorAction Stop
                Write-Host "  + ExclusionPath $p"
            } catch {
                Write-Warning "  Failed ExclusionPath ${p}: $_"
            }
        }
        try {
            Add-MpPreference -ExclusionProcess "python.exe" -ErrorAction Stop
            Write-Host "  + ExclusionProcess python.exe"
        } catch {
            Write-Warning "  Failed ExclusionProcess python.exe: $_"
        }
    }
} else {
    Write-Host "[Step 3 / E.1] skipped (-SkipDefender)"
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 4 / Sec E.2 -- Ultimate Performance + core parking
# ---------------------------------------------------------------------------
if (-not $SkipPower) {
    Write-Host "[Step 4 / E.2] Ultimate Performance power plan + core parking"
    if (-not (Test-IsAdmin)) {
        Write-Warning (
            "Not elevated - skipping powercfg. " +
            "Re-run as Administrator, or pass -SkipPower."
        )
    } else {
        # Duplicate Ultimate Performance if missing, then activate it.
        $ultimateGuid = "e9a42b02-d5df-448d-aa00-03f14749eb61"
        $listed = & powercfg.exe /list 2>&1 | Out-String
        if ($listed -notmatch $ultimateGuid) {
            Write-Host "  Duplicating Ultimate Performance scheme..."
            & powercfg.exe /duplicatescheme $ultimateGuid | Out-Null
        }
        # Activate the Ultimate Performance GUID (or the duplicate's GUID if
        # Windows assigned a new one - fall back to matching by name).
        $activeSet = $false
        try {
            & powercfg.exe /setactive $ultimateGuid 2>$null
            if ($LASTEXITCODE -eq 0) { $activeSet = $true }
        } catch {
            $activeSet = $false
        }
        if (-not $activeSet) {
            $match = (
                & powercfg.exe /list
            ) | Select-String -Pattern "Ultimate Performance"
            if ($match) {
                if ($match.Line -match "([0-9a-fA-F\-]{36})") {
                    & powercfg.exe /setactive $Matches[1] | Out-Null
                    $activeSet = $true
                    Write-Host "  Activated scheme $($Matches[1])"
                }
            }
        } else {
            Write-Host "  Activated Ultimate Performance ($ultimateGuid)"
        }

        # Disable core parking: keep 100% of cores unparked on AC power.
        & powercfg.exe /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR CPMINCORES 100
        & powercfg.exe /setactive SCHEME_CURRENT | Out-Null
        Write-Host "  CPMINCORES=100 (core parking disabled on AC)"
    }
} else {
    Write-Host "[Step 4 / E.2] skipped (-SkipPower)"
}
Write-Host ""

# ---------------------------------------------------------------------------
# Steps 16-18 / Sec E.3 -- CUDA_LAUNCH_BLOCKING must stay unset / 0
# ---------------------------------------------------------------------------
Write-Host "[Steps 16-18 / E.3] CUDA async launch (CUDA_LAUNCH_BLOCKING)"
$blocking = [Environment]::GetEnvironmentVariable(
    "CUDA_LAUNCH_BLOCKING", "Process"
)
if ($blocking -and $blocking -ne "0") {
    Write-Warning (
        "CUDA_LAUNCH_BLOCKING=$blocking is set - removing for this session " +
        "(debug flag serialises every kernel launch)."
    )
    Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue
}
# Explicitly pin to 0 in-session so child python processes inherit async mode.
Set-SessionEnv -Name "CUDA_LAUNCH_BLOCKING" -Value "0"

# Clear any stale User-level debug flag when persisting.
if ($PersistEnv) {
    $userBlocking = [Environment]::GetEnvironmentVariable(
        "CUDA_LAUNCH_BLOCKING", "User"
    )
    if ($userBlocking -and $userBlocking -ne "0") {
        Write-Warning "Clearing User-level CUDA_LAUNCH_BLOCKING=$userBlocking"
        [Environment]::SetEnvironmentVariable(
            "CUDA_LAUNCH_BLOCKING", "0", "User"
        )
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Sec C -- thread caps (avoids oversubscription across grid subprocesses)
# ---------------------------------------------------------------------------
Write-Host "[Sec C] CPU thread caps (OMP / MKL / PACER_NUM_THREADS)"
$t = [string]$NumThreads
Set-SessionEnv -Name "OMP_NUM_THREADS" -Value $t
Set-SessionEnv -Name "MKL_NUM_THREADS" -Value $t
Set-SessionEnv -Name "PACER_NUM_THREADS" -Value $t
Write-Host ""

# ---------------------------------------------------------------------------
# Sec E.4 -- Nsight Systems template (informational)
# ---------------------------------------------------------------------------
Write-Host "[Sec E.4] Nsight Systems profile template (optional)"
$nsys = Get-Command nsys -ErrorAction SilentlyContinue
if ($nsys) {
    Write-Host "  nsys found at $($nsys.Source)"
    Write-Host (
        "  Example:`n" +
        "    nsys profile --gpu-metrics-device=all --output=profile ``n" +
        "      python main_tercile.py --dataset Clothing ..."
    )
} else {
    Write-Host (
        "  nsys not on PATH. Install NVIDIA Nsight Systems, then:`n" +
        "    nsys profile --gpu-metrics-device=all --output=profile ``n" +
        "      python main_tercile.py ..."
    )
}
Write-Host ""

# ---------------------------------------------------------------------------
# Summary of CUDA-related env
# ---------------------------------------------------------------------------
Write-Host "[Check] CUDA-related environment variables:"
Get-ChildItem Env: |
    Where-Object { $_.Name -like "*CUDA*" } |
    ForEach-Object { Write-Host ("  {0}={1}" -f $_.Name, $_.Value) }

Write-Host ""
Write-Host "Done. Launch training from this shell so session env vars apply."
Write-Host (
    "Code-side speedups (TF32 / fused Adam / inference_mode / GPU sample) " +
    "live in train.py + load_data.py - no further action required."
)
