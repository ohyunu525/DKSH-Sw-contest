$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$gait = 'outputs/cad6_ga_tripod.json'
try {
    & .\run_cad6_hybrid.ps1 -Mode search -Profile $gait -NumEnvs 4 -AllowProvisionalCad *> 'hybrid_search_4.out.log'
    if ($LASTEXITCODE -ne 0) { throw "Search exited with code $LASTEXITCODE" }
    & .\run_cad6_hybrid.ps1 -Mode train -Profile $gait -NumEnvs 4 -MaxIterations 2000 -AllowProvisionalCad *> 'hybrid_train_4.out.log'
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} catch {
    $_ | Out-String | Set-Content -LiteralPath 'hybrid_search_train_failure.log' -Encoding utf8
    exit 1
}
