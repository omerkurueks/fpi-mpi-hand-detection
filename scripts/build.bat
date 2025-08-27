@echo off
REM Build and deployment script for Windows
REM This is the Windows equivalent of build.sh

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_NAME=fpi-mpi-hand-detection
set VERSION=v0.1.0
set BUILD_DATE=%date% %time%
set REGISTRY=localhost:5000

REM Colors (for PowerShell)
set GREEN=[32m
set YELLOW=[33m
set RED=[31m
set NC=[0m

goto :main

:log_info
echo [INFO] %~1
goto :eof

:log_warn
echo [WARN] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:show_help
echo.
echo 🔍 Hand-Object Inspection Detection - Build ^& Deploy Script (Windows)
echo.
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   setup           Setup development environment
echo   build           Build Docker image
echo   test            Run tests
echo   deploy-local    Deploy locally with docker-compose
echo   clean           Clean build artifacts
echo   help            Show this help
echo.
echo Options:
echo   --push          Push Docker image to registry
echo   --no-cache      Build without Docker cache
echo   --gpu           Enable GPU support
echo   --dev           Development mode
echo.
echo Examples:
echo   %~nx0 setup                    # Setup development environment
echo   %~nx0 build --push             # Build and push Docker image
echo   %~nx0 deploy-local --gpu       # Deploy locally with GPU support
echo   %~nx0 test                     # Run all tests
echo.
goto :eof

:check_dependencies
call :log_info "Checking dependencies..."

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not installed!"
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker Compose is not installed!"
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed!"
    exit /b 1
)

call :log_info "All dependencies are available ✅"
goto :eof

:setup_environment
call :log_info "Setting up development environment..."

REM Create virtual environment
if not exist "venv" (
    call :log_info "Creating virtual environment..."
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
call :log_info "Installing Python dependencies..."
pip install -r requirements.txt

if exist "requirements-dev.txt" (
    pip install -r requirements-dev.txt
)

REM Create directories
call :log_info "Creating project directories..."
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\events" mkdir data\events
if not exist "data\logs" mkdir data\logs
if not exist "data\models" mkdir data\models
if not exist "runs" mkdir runs
if not exist "runs\train" mkdir runs\train
if not exist "runs\val" mkdir runs\val
if not exist "runs\test" mkdir runs\test

REM Copy example configs
if not exist "configs\logic.yaml" (
    if exist "configs\logic.yaml.example" (
        call :log_info "Copying example configuration..."
        copy "configs\logic.yaml.example" "configs\logic.yaml"
    )
)

if not exist "configs\model.yaml" (
    if exist "configs\model.yaml.example" (
        copy "configs\model.yaml.example" "configs\model.yaml"
    )
)

call :log_info "Development environment setup complete! ✅"
call :log_info "Activate with: venv\Scripts\activate.bat"
goto :eof

:build_docker
set push=false
set no_cache=false

REM Parse options
:parse_build_args
if "%~1"=="--push" (
    set push=true
    shift /1
    goto :parse_build_args
)
if "%~1"=="--no-cache" (
    set no_cache=true
    shift /1
    goto :parse_build_args
)
if not "%~1"=="" (
    shift /1
    goto :parse_build_args
)

call :log_info "Building Docker image..."

REM Build arguments
set build_args=--build-arg VERSION=%VERSION% --build-arg BUILD_DATE="%BUILD_DATE%"

if "%no_cache%"=="true" (
    set build_args=%build_args% --no-cache
)

REM Build image
docker build %build_args% -t %PROJECT_NAME%:%VERSION% -t %PROJECT_NAME%:latest .

REM Tag for registry
if "%push%"=="true" (
    call :log_info "Tagging for registry..."
    docker tag %PROJECT_NAME%:%VERSION% %REGISTRY%/%PROJECT_NAME%:%VERSION%
    docker tag %PROJECT_NAME%:latest %REGISTRY%/%PROJECT_NAME%:latest
    
    call :log_info "Pushing to registry..."
    docker push %REGISTRY%/%PROJECT_NAME%:%VERSION%
    docker push %REGISTRY%/%PROJECT_NAME%:latest
)

call :log_info "Docker build complete! ✅"
docker images | findstr %PROJECT_NAME%
goto :eof

:run_tests
call :log_info "Running tests..."

REM Activate virtual environment if it exists
if exist "venv" (
    call venv\Scripts\activate.bat
)

REM Run pytest
pytest --version >nul 2>&1
if not errorlevel 1 (
    call :log_info "Running unit tests..."
    pytest tests/ -v --cov=src --cov-report=html --cov-report=term
    
    call :log_info "Running linting..."
    flake8 --version >nul 2>&1
    if not errorlevel 1 (
        flake8 src/ tests/
    )
    
    mypy --version >nul 2>&1
    if not errorlevel 1 (
        mypy src/
    )
    
    call :log_info "Tests completed! ✅"
) else (
    call :log_warn "pytest not found, skipping tests"
)
goto :eof

:deploy_local
set gpu=false
set dev=false

REM Parse options
:parse_deploy_args
if "%~1"=="--gpu" (
    set gpu=true
    shift /1
    goto :parse_deploy_args
)
if "%~1"=="--dev" (
    set dev=true
    shift /1
    goto :parse_deploy_args
)
if not "%~1"=="" (
    shift /1
    goto :parse_deploy_args
)

call :log_info "Deploying locally with docker-compose..."

REM Prepare environment file
echo PROJECT_NAME=%PROJECT_NAME%> .env
echo VERSION=%VERSION%>> .env
echo COMPOSE_PROJECT_NAME=%PROJECT_NAME%>> .env

REM Start services
call :log_info "Starting services..."
docker-compose up -d

REM Wait for services
call :log_info "Waiting for services to be ready..."
timeout /t 10 /nobreak >nul

REM Check health
curl -f http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    call :log_info "API service is healthy! ✅"
    call :log_info "API documentation: http://localhost:8000/docs"
) else (
    call :log_warn "API service health check failed"
)

call :log_info "Local deployment complete! ✅"
goto :eof

:clean_artifacts
call :log_info "Cleaning build artifacts..."

REM Stop and remove containers
docker-compose down --remove-orphans >nul 2>&1

REM Remove Docker images
docker rmi %PROJECT_NAME%:%VERSION% >nul 2>&1
docker rmi %PROJECT_NAME%:latest >nul 2>&1

REM Clean Python cache
for /r %%i in (*.pyc) do del "%%i" >nul 2>&1
for /d /r %%i in (__pycache__) do rmdir /s /q "%%i" >nul 2>&1
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
if exist "htmlcov" rmdir /s /q "htmlcov"
if exist ".coverage" del ".coverage"

REM Clean build directories
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

call :log_info "Cleanup complete! ✅"
goto :eof

:show_version
echo.
echo 🔍 Hand-Object Inspection Detection
echo Version: %VERSION%
echo Build Date: %BUILD_DATE%
echo.
goto :eof

:main
if "%~1"=="" goto :help
if "%~1"=="setup" goto :setup
if "%~1"=="build" goto :build
if "%~1"=="test" goto :test
if "%~1"=="deploy-local" goto :deploy_local
if "%~1"=="clean" goto :clean
if "%~1"=="version" goto :version
if "%~1"=="help" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

call :log_error "Unknown command: %~1"
goto :help

:setup
call :check_dependencies
call :setup_environment
goto :eof

:build
shift /1
call :check_dependencies
call :build_docker %*
goto :eof

:test
call :run_tests
goto :eof

:deploy_local
shift /1
call :check_dependencies
call :deploy_local %*
goto :eof

:clean
call :clean_artifacts
goto :eof

:version
call :show_version
goto :eof

:help
call :show_help
goto :eof
