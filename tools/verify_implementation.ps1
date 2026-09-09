param(
    [string]$PythonExe = 'D:\Anaconda\envs\dual-visual\python.exe',
    [string]$ProjectRoot = (Join-Path $PSScriptRoot '..')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$resolvedRoot = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedRoot 'tools\run_cli.py'
$pytestBaseTemp = Join-Path $resolvedRoot ("outputs\pytest-verification-{0}" -f [Guid]::NewGuid().ToString('N'))

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python 可执行文件不存在：$PythonExe"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "命令入口不存在：$runner"
}

Push-Location $resolvedRoot
try {
    & $PythonExe $runner verify
    if ($LASTEXITCODE -ne 0) { throw "冻结清单核查失败，退出码：$LASTEXITCODE" }
    & $PythonExe $runner --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "命令行入口核查失败，退出码：$LASTEXITCODE" }
    & $PythonExe -m compileall -q src tests tools
    if ($LASTEXITCODE -ne 0) { throw "源码字节码编译失败，退出码：$LASTEXITCODE" }
    # 某些受限 Windows 环境无法枚举系统临时目录；显式使用项目内的唯一可写目录。
    & $PythonExe -m pytest -q "--basetemp=$pytestBaseTemp"
    if ($LASTEXITCODE -ne 0) { throw "单元测试失败，退出码：$LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Output '实现核查通过：命令入口、字节码编译、可执行配置、冻结图片与全部单元测试均正常。'
