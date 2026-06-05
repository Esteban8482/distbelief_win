@echo off
REM Script rapido para probar el entrenamiento en Windows
REM Usa configuracion reducida para verificar que todo funciona

echo ===========================================
echo  DistBelief - Entrenamiento Rapido (Debug)
echo ===========================================
echo.

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
)

echo Iniciando entrenamiento de prueba...
echo Configuracion: 2 replicas, 1 shard, 2 epochs
echo.

python train.py --debug

echo.
echo ===========================================
echo  Entrenamiento completado
echo ===========================================

pause
