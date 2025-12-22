if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Error "poetry is not installed; install it first to export requirements."; exit 1
}
Write-Host "Exporting requirements.txt from Poetry lock..."
poetry export -f requirements.txt --without-hashes -o requirements.txt
Write-Host "Exported requirements.txt"