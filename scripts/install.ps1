# PowerShell install script for Windows
# Usage: .\scripts\install.ps1 -method pip
param(
    [ValidateSet("pip","poetry")] [string]$method = "pip",
    [string]$venvPath = "venv"
)

Write-Host "Install method: $method"

if ($method -eq 'pip') {
    if (-Not (Test-Path $venvPath)) {
        python -m venv $venvPath
    }
    .\$venvPath\Scripts\Activate.ps1
    if (Test-Path requirements.txt) {
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    } else {
        Write-Host "requirements.txt not found. Trying to generate from Poetry..."
        if (Get-Command poetry -ErrorAction SilentlyContinue) {
            poetry export -f requirements.txt --without-hashes -o requirements.txt
            python -m pip install --upgrade pip
            pip install -r requirements.txt
        } else {
            throw "requirements.txt missing and poetry not available. Install poetry or add requirements.txt."
        }
    }
} else {
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        poetry install
    } else {
        throw "poetry not found. Install poetry or use method pip."
    }
}

Write-Host "Install completed."