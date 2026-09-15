param(
    [Parameter(Mandatory = $true)][string]$NodeExe,
    [Parameter(Mandatory = $true)][string]$OutLib
)

$ErrorActionPreference = "Stop"

$defPath = [System.IO.Path]::ChangeExtension($OutLib, ".def")
$names = & objdump -p $NodeExe |
    Select-String -Pattern '\[\s*\d+\][^\r\n]*?\b(napi_\w+)\s*$' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } |
    Select-Object -Unique

if ($names.Count -eq 0) {
    throw "no napi exports found in $NodeExe"
}

$lines = @("LIBRARY node.exe", "EXPORTS") + $names
Set-Content -LiteralPath $defPath -Value $lines -Encoding Ascii
& dlltool -d $defPath -l $OutLib --dllname node.exe
if ($LASTEXITCODE -ne 0) {
    throw "dlltool failed"
}
