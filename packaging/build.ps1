# Build the standalone laipro desktop app (Windows).
# Run from the project root with the venv active:
#     .\packaging\build.ps1

Write-Host "Installing build + GUI dependencies..." -ForegroundColor Cyan
pip install -e ".[gui]"
pip install pyinstaller

Write-Host "Building the standalone app (this takes several minutes)..." -ForegroundColor Cyan
pyinstaller packaging/laipro.spec --noconfirm --clean

if (Test-Path "dist/laipro/laipro.exe") {
    Write-Host "`nBuilt: dist/laipro/laipro.exe" -ForegroundColor Green
    Write-Host "Ship the ENTIRE 'dist/laipro' folder (zip it) to colleagues." -ForegroundColor Green
    Write-Host "They run laipro.exe - no Python required." -ForegroundColor Green
} else {
    Write-Host "`nBuild did not produce the exe. See packaging/README.md troubleshooting." -ForegroundColor Yellow
}
