@echo off
setlocal

set SPEC=%~dp0tcm_proc.spec
set UPX_DIR=C:\Programs\_catalog\unpack\UniExtract\bin

pixi run -e noh5-tcm pyinstaller --noconfirm %SPEC%

echo.
echo Build complete: dist\tcm_proc\tcm_proc.exe
echo.
endlocal
