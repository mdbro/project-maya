# Runs as the MayaHostControl scheduled task, elevated. It reads one request written from WSL, runs only the allowlisted
# operations below with validated values, writes a result and a log line, and deletes the request. No request text ever
# reaches a shell: every program and argument is fixed here, and numbers are checked against fixed ranges.
#
# Request JSON: {"id": "letters-digits-dashes", "actions": [ ... up to 8 ... ]}
#   {"op": "query"}                                              GPU clocks, power and limits
#   {"op": "lock-graphics", "gpu": "<uuid>", "min": N, "max": N}  nvidia-smi -lgc within the GPU's supported range
#   {"op": "reset-graphics", "gpu": "<uuid>"}                     nvidia-smi -rgc
#   {"op": "lock-memory", "gpu": "<uuid>", "min": N, "max": N}    nvidia-smi -lmc, supported memory clocks only
#   {"op": "reset-memory", "gpu": "<uuid>"}                       nvidia-smi -rmc
#   {"op": "power-limit", "gpu": "<uuid>", "watts": N}            nvidia-smi -pl, never above the user's own setting
#   {"op": "restore-power-limit", "gpu": "<uuid>"}                back to the user's setting, not the higher default
#   {"op": "mount-transfer"}                                      wsl --mount of the one transfer VHD
#   {"op": "power-plan-query"} / {"op": "power-plan", "scheme": "balanced" | "high-performance"}
#   {"op": "hags-query"} / {"op": "hags", "mode": "on" | "off"}   hardware GPU scheduling; takes effect after a reboot
#   {"op": "trace-start", "profile": "GPU" | "CPU"} / {"op": "trace-stop"} / {"op": "trace-cancel"}   WPR, built-in profiles
#   {"op": "monitor-start"} / {"op": "monitor-stop"}            fixed performance counters, 1 s, CSV in the request folder
#   {"op": "host-config"}                                        memory modules, processor, GPUs, OS: read only
#   {"op": "gpu-details"}                                        nvidia-smi -q -d PERFORMANCE,CLOCK,POWER: read only
#   {"op": "vm-priority-query"} / {"op": "vm-priority", "level": "normal" | "above-normal" | "high"}   WSL VM process
#   {"op": "vm-affinity-query"} / {"op": "vm-affinity", "set": <a name from $AffinitySets>}   which logical processors the WSL
#       VM's virtual processors may run on; "all" restores the default. A new VM process starts with "all" again.
param([Parameter(Mandatory = $true)][string]$RequestDir)
$ErrorActionPreference = 'Stop'
$System = Join-Path $env:SystemRoot 'System32'
$Smi = Join-Path $System 'nvidia-smi.exe'
# Read on this host, 2026-11-07, driver 617.42: supported clocks from nvidia-smi -q -d SUPPORTED_CLOCKS, and the power
# limits as the user had set them (5090 425 W of default 575 W; 3090 390 W of default 420 W). Power is never raised above them.
$Gpus = @{
    'GPU-ea52aedc-8ee4-8e13-7ee7-c1bb774e12e9' = @{ Name = 'RTX 5090'; GraphicsMin = 180; GraphicsMax = 3090; Memory = @(14001, 13801, 7001, 810, 405); PowerMin = 400; PowerUser = 425 }
    'GPU-ea9ac138-d20c-c5f7-38b0-6d18cccabbf1' = @{ Name = 'RTX 3090'; GraphicsMin = 180; GraphicsMax = 2100; Memory = @(9751, 9501, 5001, 810, 405); PowerMin = 100; PowerUser = 390 }
}
$TransferVhd = 'E:\wsl\transfer.vhdx'
$Schemes = @{ 'balanced' = 'SCHEME_BALANCED'; 'high-performance' = 'SCHEME_MIN' }
$HagsKey = 'HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
$HagsModes = @{ 'on' = '2'; 'off' = '1' }
$TraceProfiles = @('GPU', 'CPU')
$Priorities = @{ 'normal' = 'Normal'; 'above-normal' = 'AboveNormal'; 'high' = 'High' }   # never RealTime
$VmProcesses = @('vmmemWSL', 'vmmem')
# Affinity masks for this host's 32 logical processors (Ryzen 9 9950X3D: Windows numbers the two hyperthreads of core c as
# logical processors 2c and 2c+1; cores 0-7 form one CCD, 8-15 the other). Each set keeps at least 16 logical processors.
# The L suffix keeps the literals 64-bit: a bare 0xFFFFFFFF is -1 in PowerShell.
$AffinityProcessors = 32
$AffinitySets = @{
    'all'              = 0xFFFFFFFFL   # every logical processor: the default and the restore value
    'lp0-15'           = 0x0000FFFFL   # cores 0-7, the first CCD, both hyperthreads
    'lp16-31'          = 0xFFFF0000L   # cores 8-15, the second CCD, both hyperthreads
    'even'             = 0x55555555L   # one hyperthread of every core, both CCDs
    'lp0-15-even16-31' = 0x5555FFFFL   # the first CCD whole, one hyperthread of each core on the second
    'even0-15-lp16-31' = 0xFFFF5555L   # one hyperthread of each core on the first CCD, the second whole
}
$MonitorName = 'MayaHostMonitor'
$MonitorCounters = @(
    '\Processor Information(_Total)\% Processor Performance',
    '\Processor Information(_Total)\% Processor Utility',
    '\Processor Information(_Total)\Processor Frequency',
    '\Processor Information(*)\% Processor Time',
    '\Hyper-V Hypervisor Logical Processor(_Total)\% Total Run Time',
    '\Hyper-V Hypervisor Virtual Processor(*)\% Total Run Time',
    '\Hyper-V Hypervisor Virtual Processor(*)\CPU Wait Time Per Dispatch',
    '\Memory\Available MBytes',
    '\Memory\Pages/sec',
    '\Memory\Cache Bytes',
    '\System\Processor Queue Length',
    '\System\Context Switches/sec',
    '\GPU Engine(*)\Utilization Percentage',
    '\GPU Adapter Memory(*)\Dedicated Usage',
    '\PhysicalDisk(_Total)\Disk Bytes/sec')
$RequestPath = Join-Path $RequestDir 'request.json'
$LogPath = Join-Path $RequestDir 'log.jsonl'

function Get-HostConfig {
    $memory = Get-CimInstance Win32_PhysicalMemory | Select-Object BankLabel, DeviceLocator, Capacity, Speed, ConfiguredClockSpeed, ConfiguredVoltage, Manufacturer, PartNumber
    $cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, CurrentClockSpeed, L2CacheSize, L3CacheSize
    $gpu = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate, PNPDeviceID
    $os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, TotalVisibleMemorySize, FreePhysicalMemory
    $board = Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer, Product, Version
    $bios = Get-CimInstance Win32_BIOS | Select-Object Manufacturer, SMBIOSBIOSVersion, ReleaseDate
    return [ordered]@{ program = 'cim'; args = @('host-config'); exit = 0; output = ([ordered]@{ memory = $memory; processor = $cpu; gpus = $gpu; os = $os; board = $board; bios = $bios } | ConvertTo-Json -Depth 4) }
}

function Get-VmProcess {
    $p = Get-Process -Name $VmProcesses -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { throw 'no WSL VM process found' }
    return $p
}

function Invoke-Fixed([string]$Program, [string[]]$Arguments) {
    # Native stderr must not become a terminating error; exit code and text are reported instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $text = & $Program @Arguments 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    return [ordered]@{ program = (Split-Path -Leaf $Program); args = $Arguments; exit = $code; output = $text.Trim() }
}

function Get-GpuState {
    return Invoke-Fixed $Smi @('--query-gpu=index,uuid,name,pstate,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active', '--format=csv')
}

function Get-Gpu($a) {
    $uuid = [string]$a.gpu
    if (-not $Gpus.ContainsKey($uuid)) { throw "unknown GPU $uuid" }
    return $uuid, $Gpus[$uuid]
}

$result = [ordered]@{ id = $null; started = (Get-Date).ToString('o'); ok = $false; actions = @(); state = $null; error = $null }
try {
    if (-not (Test-Path -LiteralPath $RequestPath)) { throw 'no request' }
    $raw = Get-Content -LiteralPath $RequestPath -Raw
    if ($raw.Length -gt 4096) { throw 'request too large' }
    $request = $raw | ConvertFrom-Json
    $id = [string]$request.id
    if ($id -notmatch '^[A-Za-z0-9-]{1,48}$') { throw 'invalid request id' }
    $result.id = $id
    $actions = @($request.actions)
    if ($actions.Count -lt 1 -or $actions.Count -gt 8) { throw 'a request holds 1 to 8 actions' }
    foreach ($a in $actions) {
        $op = [string]$a.op
        switch ($op) {
            'query' { $result.actions += Get-GpuState }
            'lock-graphics' {
                $uuid, $g = Get-Gpu $a
                $lo = [int]$a.min; $hi = [int]$a.max
                if ($lo -lt $g.GraphicsMin -or $hi -gt $g.GraphicsMax -or $lo -gt $hi) { throw "graphics clocks $lo,$hi outside $($g.GraphicsMin)-$($g.GraphicsMax) MHz for $($g.Name)" }
                $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-lgc', "$lo,$hi")
            }
            'reset-graphics' { $uuid, $g = Get-Gpu $a; $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-rgc') }
            'lock-memory' {
                $uuid, $g = Get-Gpu $a
                $lo = [int]$a.min; $hi = [int]$a.max
                if (-not ($g.Memory -contains $lo) -or -not ($g.Memory -contains $hi) -or $lo -gt $hi) { throw "memory clocks $lo,$hi not supported by $($g.Name)" }
                $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-lmc', "$lo,$hi")
            }
            'reset-memory' { $uuid, $g = Get-Gpu $a; $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-rmc') }
            'power-limit' {
                $uuid, $g = Get-Gpu $a
                $w = [int]$a.watts
                if ($w -lt $g.PowerMin -or $w -gt $g.PowerUser) { throw "power limit $w W outside $($g.PowerMin)-$($g.PowerUser) W for $($g.Name)" }
                $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-pl', "$w")
            }
            'restore-power-limit' { $uuid, $g = Get-Gpu $a; $result.actions += Invoke-Fixed $Smi @('-i', $uuid, '-pl', "$($g.PowerUser)") }
            'mount-transfer' { $result.actions += Invoke-Fixed (Join-Path $System 'wsl.exe') @('--mount', '--vhd', $TransferVhd, '--name', 'transfer') }
            'power-plan-query' { $result.actions += Invoke-Fixed (Join-Path $System 'powercfg.exe') @('/getactivescheme') }
            'power-plan' {
                $scheme = [string]$a.scheme
                if (-not $Schemes.ContainsKey($scheme)) { throw "unknown power scheme $scheme" }
                $result.actions += Invoke-Fixed (Join-Path $System 'powercfg.exe') @('/setactive', $Schemes[$scheme])
            }
            'hags-query' { $result.actions += Invoke-Fixed (Join-Path $System 'reg.exe') @('query', $HagsKey, '/v', 'HwSchMode') }
            'hags' {
                $mode = [string]$a.mode
                if (-not $HagsModes.ContainsKey($mode)) { throw "unknown scheduling mode $mode" }
                $result.actions += Invoke-Fixed (Join-Path $System 'reg.exe') @('add', $HagsKey, '/v', 'HwSchMode', '/t', 'REG_DWORD', '/d', $HagsModes[$mode], '/f')
                $result.reboot_required = $true
            }
            'trace-start' {
                $traceProfile = [string]$a.profile
                if ($TraceProfiles -notcontains $traceProfile) { throw "unknown trace profile $traceProfile" }
                $result.actions += Invoke-Fixed (Join-Path $System 'wpr.exe') @('-start', $traceProfile, '-filemode')
            }
            'trace-stop' { $result.actions += Invoke-Fixed (Join-Path $System 'wpr.exe') @('-stop', (Join-Path $RequestDir "trace-$id.etl")) }
            'trace-cancel' { $result.actions += Invoke-Fixed (Join-Path $System 'wpr.exe') @('-cancel') }
            'monitor-start' {
                $logman = Join-Path $System 'logman.exe'
                $null = Invoke-Fixed $logman @('delete', $MonitorName)   # an earlier definition, if any
                $result.actions += Invoke-Fixed $logman (@('create', 'counter', $MonitorName, '-c') + $MonitorCounters + @('-si', '1', '-f', 'csv', '-o', (Join-Path $RequestDir "monitor-$id"), '-ow'))
                $result.actions += Invoke-Fixed $logman @('start', $MonitorName)
            }
            'monitor-stop' { $result.actions += Invoke-Fixed (Join-Path $System 'logman.exe') @('stop', $MonitorName) }
            'host-config' { $result.actions += Get-HostConfig }
            'gpu-details' { $result.actions += Invoke-Fixed $Smi @('-q', '-d', 'PERFORMANCE,CLOCK,POWER') }
            'vm-priority-query' {
                $p = Get-VmProcess
                $result.actions += [ordered]@{ program = 'process'; args = @('priority-query'); exit = 0; output = "$($p.ProcessName) $($p.Id) $($p.PriorityClass)" }
            }
            'vm-priority' {
                $level = [string]$a.level
                if (-not $Priorities.ContainsKey($level)) { throw "unknown priority $level" }
                $p = Get-VmProcess
                $before = "$($p.PriorityClass)"
                $p.PriorityClass = $Priorities[$level]
                $p.Refresh()
                $result.actions += [ordered]@{ program = 'process'; args = @('priority', $level); exit = 0; output = "$($p.ProcessName) $($p.Id) $before -> $($p.PriorityClass)" }
            }
            'vm-affinity-query' {
                $p = Get-VmProcess
                # Process affinity steers virtual processors only under the root scheduler (type 0x4), the client default.
                $scheduler = try { (Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Hyper-V-Hypervisor'; Id = 2 } -MaxEvents 1).Message } catch { 'scheduler event not found' }
                $line = '{0} {1} affinity 0x{2:X8} of {3} logical processors; {4} threads; {5}' -f $p.ProcessName, $p.Id, $p.ProcessorAffinity.ToInt64(), [Environment]::ProcessorCount, $p.Threads.Count, $scheduler
                $result.actions += [ordered]@{ program = 'process'; args = @('affinity-query'); exit = 0; output = $line }
            }
            'vm-affinity' {
                $set = [string]$a.set
                if (-not $AffinitySets.ContainsKey($set)) { throw "unknown affinity set $set" }
                if ([Environment]::ProcessorCount -ne $AffinityProcessors) { throw "the affinity sets are for $AffinityProcessors logical processors, this host has $([Environment]::ProcessorCount)" }
                $p = Get-VmProcess
                $before = $p.ProcessorAffinity.ToInt64()
                $p.ProcessorAffinity = [IntPtr][long]$AffinitySets[$set]
                $p.Refresh()
                $line = '{0} {1} 0x{2:X8} -> 0x{3:X8}' -f $p.ProcessName, $p.Id, $before, $p.ProcessorAffinity.ToInt64()
                $result.actions += [ordered]@{ program = 'process'; args = @('affinity', $set); exit = 0; output = $line }
            }
            default { throw "unknown op $op" }
        }
    }
    $result.state = Get-GpuState
    $result.ok = -not ($result.actions | Where-Object { $_.exit -ne 0 })
}
catch { $result.error = $_.Exception.Message }
finally {
    $result.finished = (Get-Date).ToString('o')
    $name = if ($result.id) { "result-$($result.id).json" } else { 'result-invalid.json' }
    ($result | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $RequestDir $name) -Encoding UTF8
    ($result | ConvertTo-Json -Depth 6 -Compress) | Add-Content -LiteralPath $LogPath -Encoding UTF8
    Remove-Item -LiteralPath $RequestPath -ErrorAction SilentlyContinue
}
