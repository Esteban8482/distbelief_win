@echo off
REM Script de lanzamiento para cluster DistBelief en Windows
REM 
REM Uso:
REM   launch_cluster.bat ps       - Lanzar Parameter Server
REM   launch_cluster.bat replica  - Lanzar réplicas (requiere PS_HOST)
REM
REM Variables de entorno:
REM   PS_HOST        - IP del Parameter Server (ej: 192.168.1.10)
REM   PS_PORT_BASE   - Puerto base (default: 29500)
REM   NUM_REPLICAS   - Número de réplicas en este nodo (default: 4)
REM   NUM_SHARDS     - Número de shards del PS (default: 1)

echo ===========================================
echo  DistBelief - Lanzador de Cluster
echo ===========================================
echo.

if "%1"=="" (
    echo Uso: launch_cluster.bat [ps ^| replica]
    echo.
    echo Ejemplos:
    echo   launch_cluster.bat ps
    echo   set PS_HOST=192.168.1.10 ^&^& launch_cluster.bat replica
    goto :eof
)

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
)

if "%1"=="ps" (
    echo Lanzando Parameter Server...
    echo   Shards: %NUM_SHARDS%
    echo   Puerto base: %PS_PORT_BASE%
    echo.
    python train.py --role ps --num-shards %NUM_SHARDS% --ps-port-base %PS_PORT_BASE%
    
) else if "%1"=="replica" (
    if "%PS_HOST%"=="" (
        echo ERROR: Define PS_HOST con la IP del Parameter Server
        echo   Ejemplo: set PS_HOST=192.168.1.10
        goto :eof
    )
    
    echo Lanzando Model Replicas...
    echo   PS: %PS_HOST%:%PS_PORT_BASE%
    echo   Replicas: %NUM_REPLICAS%
    echo.
    python train.py --role replica --ps-host %PS_HOST% --ps-port-base %PS_PORT_BASE% --num-replicas %NUM_REPLICAS%
    
) else (
    echo Rol desconocido: %1
    echo Roles válidos: ps, replica
)
