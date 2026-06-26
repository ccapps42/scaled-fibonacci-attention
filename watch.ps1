Set-Location $PSScriptRoot

# Wait for the running blur sweep to finish, then launch run_all.ps1 (everything else not done).
Write-Host "$(Get-Date -Format 'MM-dd HH:mm') watching for the blur run to finish..."
while (Get-CimInstance Win32_Process -Filter "name='python.exe'" |
       Where-Object { $_.CommandLine -match 'fib_blur_stag' }) {
    Start-Sleep -Seconds 120
}
Write-Host "$(Get-Date -Format 'MM-dd HH:mm') blur done -- starting run_all.ps1"
.\run_all.ps1
Write-Host "$(Get-Date -Format 'MM-dd HH:mm') all runs finished"
