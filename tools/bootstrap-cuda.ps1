# Download a workspace-local compiler; never change the system CUDA installation.
$ErrorActionPreference = 'Stop'
$enttCudaVersion = '13.2.0'
$enttCudaRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$enttCudaCache = Join-Path $enttCudaRoot '.dev/cuda-downloads'
$enttCudaToolkit = Join-Path $enttCudaRoot ".dev/cuda-$enttCudaVersion"
$enttCudaBase = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$enttCudaManifestFile = Join-Path $enttCudaCache "redistrib_$enttCudaVersion.json"

New-Item -ItemType Directory -Force $enttCudaCache, $enttCudaToolkit | Out-Null

if(-not (Test-Path -LiteralPath $enttCudaManifestFile)) {
    Invoke-WebRequest -Uri ($enttCudaBase + "redistrib_$enttCudaVersion.json") -OutFile $enttCudaManifestFile
}

$enttCudaManifest = Get-Content -LiteralPath $enttCudaManifestFile -Raw | ConvertFrom-Json

$enttCudaParts = @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_crt', 'libnvvm')

foreach($enttCudaPart in $enttCudaParts) {
    $enttCudaPackage = $enttCudaManifest.$enttCudaPart.'windows-x86_64'
    $enttCudaFilename = Split-Path $enttCudaPackage.relative_path -Leaf
    $enttCudaArchive = Join-Path $enttCudaCache $enttCudaFilename

    if(-not (Test-Path -LiteralPath $enttCudaArchive)) {
        Write-Host "Downloading $enttCudaFilename"
        Invoke-WebRequest -Uri ($enttCudaBase + $enttCudaPackage.relative_path) -OutFile $enttCudaArchive
    }

    $enttCudaHash = (Get-FileHash -LiteralPath $enttCudaArchive -Algorithm SHA256).Hash

    if($enttCudaHash -ne $enttCudaPackage.sha256) {
        throw "Checksum mismatch: $enttCudaArchive. Remove the incomplete archive before retrying."
    }

    Write-Host "Extracting $enttCudaFilename"
    Expand-Archive -LiteralPath $enttCudaArchive -DestinationPath $enttCudaCache -Force
    $enttCudaExtracted = Join-Path $enttCudaCache ([System.IO.Path]::GetFileNameWithoutExtension($enttCudaFilename))
    Copy-Item -Path (Join-Path $enttCudaExtracted '*') -Destination $enttCudaToolkit -Recurse -Force
}

# Original package licenses are retained in cuda-downloads alongside each archive.
Write-Host "CUDA $enttCudaVersion is available at $enttCudaToolkit"
