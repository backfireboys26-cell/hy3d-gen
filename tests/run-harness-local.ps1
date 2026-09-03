<#
.SYNOPSIS
  Run the hy3d-gen contract harness on THIS Windows machine against the vault's patch - no GPU, no model.
.DESCRIPTION
  1. clones Hunyuan3D-2GP @ f2456e0 from the local checkout (default C:\ai3d\Hunyuan3D-2GP) into a
     fresh temp dir with core.autocrlf=false (the patch is LF; a CRLF working tree makes `git apply`
     refuse), 2. applies <vault>\engine\generation\patches\rsv4-stack.patch, 3. runs tests\harness.py
     with the Hunyuan venv's python (default C:\ai3d\venv) using <vault>\engine\generation\docker\auth_gate.py
     as the gate. The stub worker replaces the pipeline, so torch is imported but never touches a GPU.
  Exit code = the harness's (0 = every case PASS). Logs: <OutDir>\harness.log (+ .gate.txt, hang-child.log).
.EXAMPLE
  powershell -NoProfile -File engine\generation\tests\run-harness-local.ps1 -OutDir projects\evidence\<date>-x\harness
#>
[CmdletBinding()]
param(
    [string]$UpstreamRepo = 'C:\ai3d\Hunyuan3D-2GP',
    [string]$Python = 'C:\ai3d\venv\Scripts\python.exe',
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA ("Temp\hy3d-harness\" + (Get-Date -Format yyyyMMdd-HHmmss))),
    [string]$Ref = 'f2456e0'
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$gen = Split-Path -Parent $here                       # engine\generation
$patch = Join-Path $gen 'patches\rsv4-stack.patch'
$gate = Join-Path $gen 'docker\auth_gate.py'
foreach ($p in @($UpstreamRepo, $Python, $patch, $gate)) { if (-not (Test-Path $p)) { throw "missing: $p" } }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$work = Join-Path $env:LOCALAPPDATA ("Temp\hy3d-harness\upstream-" + (Get-Date -Format yyyyMMdd-HHmmss))
Write-Host "[harness] clone $UpstreamRepo @ $Ref -> $work"
git clone -q $UpstreamRepo $work
git -C $work config core.autocrlf false
git -C $work checkout -q $Ref
git -C $work rm -q --cached -r . ; git -C $work reset -q --hard   # renormalize the tree to LF
git -C $work apply --check $patch
git -C $work apply $patch
Write-Host "[harness] patch applied: $(git -C $work diff --stat | Select-Object -Last 1)"
$env:HY3D_API_DIR = $work; $env:HY3D_REPO = $work; $env:HY3D_GATE = $gate
$env:HY3D_HARNESS_LOG = Join-Path $OutDir 'harness.log'
& $Python (Join-Path $here 'harness.py')
$rc = $LASTEXITCODE
Write-Host "[harness] exit $rc - log $($env:HY3D_HARNESS_LOG)"
Get-Content $env:HY3D_HARNESS_LOG -Tail 1
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
exit $rc
