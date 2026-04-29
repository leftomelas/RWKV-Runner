@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "BACKEND_ROOT=%~dp0.."
for %%I in ("%BACKEND_ROOT%") do set "BACKEND_ROOT=%%~fI"
set "REPO_ROOT=%BACKEND_ROOT%\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "PYTHON_EXE=%REPO_ROOT%\py310\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [Albatross] Bundled Python not found: "%PYTHON_EXE%"
  exit /b 1
)

if not defined ALBATROSS_PYTHON310_ROOT (
  set "ALBATROSS_PYTHON310_ROOT=%LOCALAPPDATA%\Programs\Python\Python310"
)

if not defined ALBATROSS_PYTHON_INCLUDE (
  set "ALBATROSS_PYTHON_INCLUDE=%ALBATROSS_PYTHON310_ROOT%\include"
)
if not defined ALBATROSS_PYTHON_LIB_DIR (
  set "ALBATROSS_PYTHON_LIB_DIR=%ALBATROSS_PYTHON310_ROOT%\libs"
)

if not exist "%ALBATROSS_PYTHON_INCLUDE%\Python.h" (
  echo [Albatross] Python.h not found: "%ALBATROSS_PYTHON_INCLUDE%\Python.h"
  echo [Albatross] Install full Python 3.10 or set ALBATROSS_PYTHON310_ROOT.
  exit /b 1
)
if not exist "%ALBATROSS_PYTHON_LIB_DIR%\python310.lib" (
  echo [Albatross] python310.lib not found: "%ALBATROSS_PYTHON_LIB_DIR%\python310.lib"
  echo [Albatross] Install full Python 3.10 or set ALBATROSS_PYTHON310_ROOT.
  exit /b 1
)

if defined VS_VCVARS64 (
  set "VCVARS64=%VS_VCVARS64%"
)

if not defined VCVARS64 (
  for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%%I\VC\Auxiliary\Build\vcvars64.bat"
    if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set "VS_INSTALL_PATH=%%I"
  )
)

if not defined VCVARS64 (
  for %%E in (Community Professional Enterprise BuildTools) do (
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
      set "VS_INSTALL_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\%%E"
    )
  )
)

if not defined VCVARS64 (
  echo [Albatross] Visual Studio 2022 vcvars64.bat not found.
  echo [Albatross] Set VS_VCVARS64 to your vcvars64.bat path and rerun.
  exit /b 1
)

if defined ALBATROSS_ARCH (
  set "ARCH=%ALBATROSS_ARCH%"
) else (
  set "ARCH=%~1"
)
if "%ARCH%"=="" set "ARCH=auto"

echo [Albatross] VS environment: "%VCVARS64%"
echo [Albatross] Python include: "%ALBATROSS_PYTHON_INCLUDE%"
echo [Albatross] Python libs: "%ALBATROSS_PYTHON_LIB_DIR%"
echo [Albatross] Arch: %ARCH%

call "%VCVARS64%"
if errorlevel 1 exit /b %errorlevel%

if defined VS_INSTALL_PATH (
  if exist "%VS_INSTALL_PATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
    set "PATH=%VS_INSTALL_PATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
  )
)

if not defined MAX_JOBS set "MAX_JOBS=1"
echo [Albatross] MAX_JOBS: %MAX_JOBS%
echo [Albatross] Build dir: "%BACKEND_ROOT%\rwkv7_state_fwd_fp16_build"
where ninja

cd /d "%BACKEND_ROOT%"
"%PYTHON_EXE%" scripts\build_albatross_kernel.py --arch "%ARCH%"
exit /b %errorlevel%
