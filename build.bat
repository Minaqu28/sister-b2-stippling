@echo off
REM Usage:
REM   build.bat [path\to\opencv\build]        build, launch the CLI/TUI (bin\stipple.exe)
REM   build.bat gui [path\to\opencv\build]    build, launch the GUI (bin\stipple_gui.exe)
REM   build.bat clean                         delete build\ and bin\
setlocal

REM Deleting the build tree cannot live inside CMake, which would be removing
REM the files it is running from, so it belongs here.
if /i "%~1"=="clean" (
    if exist build rmdir /s /q build
    if exist bin rmdir /s /q bin
    echo Removed build\ and bin\.
    exit /b 0
)

set "TARGET=stipple"
set "OPENCV_ARG=%~1"
if /i "%~1"=="gui" (
    set "TARGET=stipple_gui"
    set "OPENCV_ARG=%~2"
)

set "OPENCV_ROOT=%OPENCV_ARG%"
if "%OPENCV_ROOT%"=="" if exist "C:\opencv\build" set "OPENCV_ROOT=C:\opencv\build"
if "%OPENCV_ROOT%"=="" if exist "%USERPROFILE%\opencv-extract\opencv\build" set "OPENCV_ROOT=%USERPROFILE%\opencv-extract\opencv\build"

where cmake >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set "PATH=%PATH%;C:\Program Files\CMake\bin"
    ) else (
        echo Error: cmake not found. Install with: winget install Kitware.CMake
        exit /b 1
    )
)

set "CMAKE_ARGS=-S . -B build -A x64"
if not "%OPENCV_ROOT%"=="" set "CMAKE_ARGS=%CMAKE_ARGS% -DOpenCV_DIR=%OPENCV_ROOT%"

echo === Configuring ===
cmake %CMAKE_ARGS% || goto :failed

echo.
echo === Building %TARGET% ===
cmake --build build --config Release --target %TARGET% || goto :failed

if not exist "bin\%TARGET%.exe" (
    echo Error: build succeeded but bin\%TARGET%.exe is missing.
    exit /b 1
)

echo.
echo === Launching bin\%TARGET%.exe ===
if /i "%TARGET%"=="stipple" echo Enter q at any prompt to quit.
echo.
bin\%TARGET%.exe
exit /b %errorlevel%

:failed
echo.
echo Build failed. Check that OpenCV is installed and, if it is in a custom
echo location, pass it:  build.bat C:\path\to\opencv\build
echo                 or: build.bat gui C:\path\to\opencv\build
exit /b 1
