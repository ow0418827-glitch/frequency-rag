param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot '..')
)

# 只读核查设计包；不运行模型，不修改原项目，不写入任何报告。
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$designRoot = [IO.Path]::GetFullPath($ProjectRoot)
$utf8Strict = [Text.UTF8Encoding]::new($false, $true)
$script:checkedHashes = 0

function Read-StrictUtf8([string]$Path) {
    return [IO.File]::ReadAllText($Path, $utf8Strict)
}

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Verify-Hash([string]$Path, [string]$Expected) {
    Require (Test-Path -LiteralPath $Path -PathType Leaf) "引用文件不存在：$Path"
    Require ($Expected -match '^[0-9a-f]{64}$') "校验值格式错误：$Path"
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    Require ($actual -eq $Expected) "引用文件已经改变：$Path"
    $script:checkedHashes += 1
}

$required = @(
    'README.md',
    'docs/设计与实现方案.md',
    'docs/实验协议与验收标准.md',
    'docs/现有项目兼容清单.md',
    'configs/design_defaults.json',
    'manifests/reference_snapshot.json',
    'tools/verify_design.ps1'
)

foreach ($relative in $required) {
    $path = Join-Path $designRoot $relative
    Require (Test-Path -LiteralPath $path -PathType Leaf) "设计文件缺失：$relative"
    $contents = Read-StrictUtf8 $path
    Require (-not [string]::IsNullOrWhiteSpace($contents)) "设计文件为空：$relative"
    if ($relative.EndsWith('.md')) {
        foreach ($match in [regex]::Matches($contents, '\]\(([^)]+)\)')) {
            $destination = $match.Groups[1].Value
            if ($destination -match '^(https?://|#)') { continue }
            $linkedPath = [IO.Path]::GetFullPath(
                (Join-Path ([IO.Path]::GetDirectoryName($path)) $destination)
            )
            Require (Test-Path -LiteralPath $linkedPath -PathType Leaf) "本地链接不存在：$relative -> $destination"
        }
    }
}

$draft = (Read-StrictUtf8 (Join-Path $designRoot 'configs/design_defaults.json')) | ConvertFrom-Json
$snapshot = (Read-StrictUtf8 (Join-Path $designRoot 'manifests/reference_snapshot.json')) | ConvertFrom-Json

Require ($draft.status -eq 'design_only_not_executable_attack') '配置必须明确属于设计阶段。'
Require ($snapshot.status -eq 'frozen_from_saved_results_not_reselected_or_rerun') '历史清单状态错误。'
Require ($snapshot.samples.Count -eq 10 -and $snapshot.num_samples -eq 10) '历史清单必须有十行。'
Require ($draft.experiment.compatibility_rows -eq $snapshot.num_samples) '配置与历史行数不一致。'
Require ($draft.reference_root -eq $snapshot.reference_root) '参考项目路径不一致。'
Require ($draft.dataset -eq $snapshot.dataset) '数据领域不一致。'
Require ($draft.evaluation.primary_encoder -eq $snapshot.encoder) '主评估器与历史不一致。'
Require ($draft.surrogates.Count -eq 3) '主攻击必须保留三个代理。'
Require ([Math]::Abs($draft.attack.epsilon - 16.0 / 255.0) -lt 1e-12) '扰动预算不一致。'
Require ([Math]::Abs($draft.attack.spatial_step_size - 0.5 / 255.0) -lt 1e-12) '空间步长不一致。'
Require (-not $draft.frequency.final_saved_image_strictly_bandlimited) '不能保证落盘图片严格限频。'
Require (-not $draft.evaluation.use_primary_evaluator_gradient) '主评估器不应参与攻击反向传播。'
Require ($null -eq $snapshot.historical_attack_time_seconds) '不能为历史结果补造耗时。'
Require ($null -eq $snapshot.historical_peak_memory_bytes) '不能为历史结果补造显存。'
Require (-not $snapshot.weight_revisions_verified) '本阶段没有验证实际权重修订。'

foreach ($file in $snapshot.reference_files) {
    Verify-Hash (Join-Path $snapshot.reference_root $file.relative_path) $file.sha256
}
Verify-Hash $snapshot.summary_path $snapshot.summary_sha256
foreach ($file in $snapshot.data_files) { Verify-Hash $file.path $file.sha256 }
foreach ($file in $snapshot.vector_files) { Verify-Hash $file.path $file.sha256 }

$sampleIds = @{}
foreach ($sample in $snapshot.samples) {
    Require (-not $sampleIds.ContainsKey($sample.sample_id)) '样本标识重复。'
    $sampleIds[$sample.sample_id] = $true
    Require ($sample.historical_steps -eq $draft.attack.reference_steps) '历史步数与对照配置不一致。'
    Require ([Math]::Abs($sample.historical_epsilon - $draft.attack.epsilon) -lt 1e-12) '历史像素预算与配置不一致。'
    Require (-not [string]::IsNullOrWhiteSpace($sample.target_text)) '目标文本缺失。'
    Require ($sample.historical_metrics.embedding_dimension -eq 512) '历史向量维度不一致。'
    Verify-Hash $sample.source_image $sample.source_sha256
    Verify-Hash $sample.target_image $sample.target_sha256
    Verify-Hash $sample.historical_adversarial_image $sample.historical_adversarial_sha256
    Verify-Hash $sample.historical_attack_metadata $sample.historical_attack_metadata_sha256
}

$sourceCount = @($snapshot.samples.source_sha256 | Select-Object -Unique).Count
$targetCount = @($snapshot.samples.target_sha256 | Select-Object -Unique).Count
Require ($sourceCount -eq 8) '历史不同源图数量已与文档不一致。'
Require ($targetCount -eq 2) '历史不同目标图数量已与文档不一致。'
$meanGain = ($snapshot.samples | ForEach-Object {
    $_.historical_metrics.cosine_metrics.target_cosine_gain
} | Measure-Object -Average).Average
Require ([Math]::Abs($meanGain - $snapshot.historical_averages.mean_target_cosine_gain) -lt 1e-10) '历史逐行指标与汇总不一致。'

Write-Output "设计包核查通过：$($required.Count) 份必需文件；$($snapshot.samples.Count) 行历史样本；$sourceCount 张不同源图；$targetCount 张不同目标图；$script:checkedHashes 项引用内容校验。"
Write-Output '以上只证明设计文件、配置与历史引用未漂移；实现状态和真实运行证据请看实现核查记录。'
