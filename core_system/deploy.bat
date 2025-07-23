@echo off
REM Intelligent Document Processing System - Windows Deployment Script
REM Adobe India Hackathon 2025

setlocal enabledelayedexpansion

echo 🚀 Deploying Intelligent Document Processing System
echo ==================================================

REM Configuration
set IMAGE_NAME=intelligent-document-processor
set CONTAINER_NAME=doc-processor
set VERSION=1.0.0

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker first.
    exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running. Please start Docker.
    exit /b 1
)

if "%1"=="build" goto :build
if "%1"=="test" goto :test
if "%1"=="run-1a" goto :run_1a
if "%1"=="run-1b" goto :run_1b
if "%1"=="demo" goto :demo
if "%1"=="cleanup" goto :cleanup
if "%1"=="info" goto :info
if "%1"=="all" goto :all
goto :help

:build
echo [INFO] Building Docker image...
docker build --platform linux/amd64 --tag %IMAGE_NAME%:%VERSION% --tag %IMAGE_NAME%:latest .
if errorlevel 1 (
    echo [ERROR] Docker build failed
    exit /b 1
)
echo [SUCCESS] Docker image built successfully
goto :end

:test
echo [INFO] Testing Docker image...
mkdir test_input test_output 2>nul
echo Dummy PDF content for testing > test_input\test.pdf
docker run --rm -v "%cd%\test_input:/app/input:ro" -v "%cd%\test_output:/app/output" --network none %IMAGE_NAME%:latest
if exist test_output\test.json (
    echo [SUCCESS] Container test passed - output file generated
) else (
    echo [WARNING] Container ran but no output file found
)
rmdir /s /q test_input test_output 2>nul
goto :end

:run_1a
set input_dir=%2
set output_dir=%3
if "%input_dir%"=="" set input_dir=.\input
if "%output_dir%"=="" set output_dir=.\output

echo [INFO] Running Challenge 1a processing...
if not exist "%input_dir%" (
    echo [ERROR] Input directory %input_dir% does not exist
    exit /b 1
)
mkdir "%output_dir%" 2>nul
docker run --rm -v "%cd%\%input_dir%:/app/input:ro" -v "%cd%\%output_dir%:/app/output" --network none --memory=16g --cpus=8 %IMAGE_NAME%:latest
echo [SUCCESS] Challenge 1a processing completed
echo [INFO] Output files saved to: %output_dir%
goto :end

:run_1b
set work_dir=%2
if "%work_dir%"=="" set work_dir=.\Challenge_1b

echo [INFO] Running Challenge 1b processing...
if not exist "%work_dir%" (
    echo [ERROR] Working directory %work_dir% does not exist
    exit /b 1
)
docker run --rm -v "%cd%\%work_dir%:/app" --network none --memory=16g --cpus=8 %IMAGE_NAME%:latest python unified_processor.py --challenge 1b
echo [SUCCESS] Challenge 1b processing completed
goto :end

:demo
echo [INFO] Running system demonstration...
docker run --rm --network none %IMAGE_NAME%:latest python demo.py
echo [SUCCESS] Demo completed
goto :end

:cleanup
echo [INFO] Cleaning up Docker resources...
docker system prune -f
echo [SUCCESS] Cleanup completed
goto :end

:info
echo [INFO] System Information:
docker --version
echo Platform: %PROCESSOR_ARCHITECTURE%
wmic computersystem get TotalPhysicalMemory /value | find "="
wmic cpu get NumberOfCores /value | find "="

echo [INFO] Image Information:
docker images %IMAGE_NAME%:latest
goto :end

:all
call :build
call :test
echo [SUCCESS] Full deployment completed successfully!
goto :end

:help
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   build               Build the Docker image
echo   test                Test the built image
echo   run-1a [input] [output]   Run Challenge 1a (default: .\input .\output)
echo   run-1b [workdir]    Run Challenge 1b (default: .\Challenge_1b)
echo   demo                Run system demonstration
echo   cleanup             Clean up Docker resources
echo   info                Show system information
echo   help                Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 run-1a .\my_pdfs .\results
echo   %0 run-1b .\Challenge_1b
echo   %0 demo

:end
endlocal
