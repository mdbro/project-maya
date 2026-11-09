# One-time setup, run from an elevated PowerShell. It copies maya-host-control.ps1 into an administrator-only folder and
# registers the on-demand task MayaHostControl, which runs that copy elevated. Afterwards the task starts without a
# prompt from an ordinary session, including WSL: schtasks.exe /Run /TN MayaHostControl
# The task survives reboots and WSL restarts. Clock locks and power limits may not; the WSL tool re-applies them on demand.
# Remove everything with:
#   Unregister-ScheduledTask -TaskName MayaHostControl -Confirm:$false; Remove-Item -Recurse "$env:ProgramData\MayaHostControl"
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run this from an elevated (Administrator) PowerShell.' }
$RequestDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # the user folder this script sits in; requests and results go here
$Target = Join-Path $env:ProgramData 'MayaHostControl'
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -LiteralPath (Join-Path $RequestDir 'maya-host-control.ps1') -Destination (Join-Path $Target 'maya-host-control.ps1') -Force
# Only Administrators and SYSTEM can change the script the elevated task runs; Users may read and run it.
& icacls.exe $Target /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-545:(OI)(CI)RX' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'could not restrict the script folder' }
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -RequestDir "{1}"' -f (Join-Path $Target 'maya-host-control.ps1'), $RequestDir
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'MayaHostControl' -Action $action -Principal $principal -Settings $settings -Force `
    -Description 'Allowlisted host operations for MiniMax benchmarking: GPU clocks and power, transfer VHD mount, power plan, GPU scheduling, GPU trace, WSL VM priority and processor affinity. Requests come from the user folder; the script lives in ProgramData.' | Out-Null
Write-Host "Registered MayaHostControl. Elevated script: $Target\maya-host-control.ps1. Requests and results: $RequestDir"
