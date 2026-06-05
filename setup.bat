@echo off
REM Script de configuración para Windows 10/11
REM Instala todas las dependencias necesarias

echo ===========================================
echo  DistBelief - Configuracion para Windows
echo ===========================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en PATH
    echo Descargar desde: https://www.python.org/downloads/
    exit /b 1
)

echo [OK] Python detectado
python --version
echo.

REM Verificar pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip no esta instalado
    exit /b 1
)

echo [OK] pip detectado
echo.

REM Crear entorno virtual (opcional)
if not exist "venv" (
    echo Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [ADVERTENCIA] No se pudo crear entorno virtual, continuando con Python global
    ) else (
        echo [OK] Entorno virtual creado
        echo Activando entorno virtual...
        call venv\Scripts\activate
    )
) else (
    echo Activando entorno virtual existente...
    call venv\Scripts\activate
)

echo.

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias
    exit /b 1
)

echo.
echo [OK] Dependencias instaladas correctamente
echo.

REM Verificar instalacion
echo Verificando instalacion...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo.
echo ===========================================
echo  Configuracion completada exitosamente
echo ===========================================
echo.
echo Para entrenar:
echo   python train.py
echo.
echo Para modo debug (rapido):
echo   python train.py --debug
echo.
echo Para ver opciones:
echo   python train.py --help
echo.

pause
