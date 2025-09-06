$envName = "notecast"

Write-Output >>> Creating environment $envName from environment.yml"
conda env create -f environment.yml -n $envName
if ($LASTEXITCODE -ne 0) {
    Write-Output ">>> Environment already exists, updating..."
    conda env update -f environment.yml --prune
}

Write-Output ">>> Activating environment"
conda activate $envName

Write-Output ">>> Download completed!"