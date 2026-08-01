@echo off
setlocal

set SPEC=%~dp0tcm_clc_txt.spec
set UPX_DIR=C:\Programs\_catalog\unpack\UniExtract\bin

pixi run -e noh5-tcm pyinstaller --noconfirm %SPEC%

echo.
echo Build complete: dist\tcm_clc_txt\tcm_clc_txt.exe
echo.
endlocal
