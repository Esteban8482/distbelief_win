# DistBelief - Implementación de Distributed Deep Learning

Implementación del framework **DistBelief** descrito en el paper [*"Large Scale Distributed Deep Networks"*](https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html) de Dean et al. (NIPS 2012), con entrenamiento en **CIFAR-10**.

## Características Implementadas

| Característica | Descripción | Paper Ref. |
|---|---|---|
| **Parameter Server Sharded** | Parámetros divididos entre múltiples shards | Sección 4.1 |
| **Downpour SGD** | SGD asíncrono con fetch/push de parámetros | Sección 4.1, Algoritmo 1 |
| **Adagrad Adaptativo** | Learning rate adaptativo por parámetro | Sección 4.1, Ecuación Adagrad |
| **Model Replicas** | Múltiples réplicas entrenando en paralelo | Sección 4.1 |
| **Warm Start** | Entrenar con 1 replica antes de activar el resto | Sección 4.1 |
| **Gradient Clipping** | Clip de gradientes para estabilidad | - |
| **Checkpointing** | Guardado periódico del estado | - |
| **Model Parallelism** | Red diseñada para particionamiento | Sección 3 |

## Arquitectura del Sistema

```
+---------------------------------------------------+
|              DistBelief Coordinator                |
|  +---------------------------------------------+  |
|  |         Parameter Server (Sharded)           |  |
|  |  +--------+  +--------+  +--------+         |  |
|  |  |Shard 0 |  |Shard 1 |  |Shard N | ...     |  |
|  |  |Params  |  |Params  |  |Params  |         |  |
|  |  |Adagrad |  |Adagrad |  |Adagrad |         |  |
|  |  +--------+  +--------+  +--------+         |  |
|  +---------------------------------------------+  |
|                      |                            |
|    +-----------------+--------------------+       |
|    |                 |                    |       |
| +--+--+  +--+--+  +--+--+  +--+--+    +--+--+   |
| |Rep.0|  |Rep.1|  |Rep.2|  |Rep.3|... |Rep.N|   |
| |Train|  |Train|  |Train|  |Train|    |Train|   |
| +--+--+  +--+--+  +--+--+  +--+--+    +--+--+   |
+----|--------|--------|--------|------------|-----+
     |        |        |        |            |
  Data 0   Data 1   Data 2   Data 3     Data N
```

## Requisitos

- **Python**: 3.7+
- **PyTorch**: 1.9+
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4GB mínimo (recomendado 8GB+)
- **CPU**: Cuantos más cores, mejor (cada replica usa 1 proceso)

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd distbelief
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision {torchvision.__version__}')"
```

## Uso

### Entrenamiento Básico

```bash
python train.py
```

Esto usa la configuración por defecto:
- 2 shards del Parameter Server
- 4 Model Replicas
- Adagrad adaptativo
- 50 epochs
- Batch size 128

### Configuración Personalizada

```bash
# 8 réplicas, 4 shards, batch size 64
python train.py --num-replicas 8 --num-shards 4 --batch-size 64

# Sin Adagrad, SGD con lr fijo
python train.py --no-adagrad --lr 0.01

# Más steps de warm start
python train.py --warm-start-steps 200 --epochs 100

# Limitar steps totales (ignora epochs)
python train.py --max-steps 5000
```

### Modo Debug (rápido, para pruebas)

```bash
python train.py --debug
```

Esto usa una configuración reducida:
- 2 réplicas, 1 shard
- Menos epochs
- Warm start corto

### Opciones Completas

```bash
python train.py --help
```

| Parámetro | Descripción | Default |
|---|---|---|
| `--num-shards` | Shards del Parameter Server | 2 |
| `--num-replicas` | Model Replicas | 4 |
| `--batch-size` | Tamaño de mini-batch | 128 |
| `--no-adagrad` | Usar SGD en vez de Adagrad | False |
| `--adagrad-gamma` | Gamma de Adagrad | 0.01 |
| `--lr` | Learning rate (sin Adagrad) | 0.001 |
| `--fetch-freq` | Frecuencia de fetch n_fetch | 1 |
| `--push-freq` | Frecuencia de push n_push | 1 |
| `--warm-start-steps` | Steps de warm start | 100 |
| `--epochs` | Número de epochs | 50 |
| `--seed` | Semilla aleatoria | 42 |
| `--data-dir` | Directorio de CIFAR-10 | ./data |
| `--output-dir` | Directorio de salida | ./output |
| `--debug` | Modo debug rápido | False |

## Estructura del Proyecto

```
distbelief/
├── train.py                  # Script principal de entrenamiento
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración del sistema
│   ├── utils.py              # Utilidades y comunicación
│   ├── network.py            # Red neuronal (CIFAR-10)
│   ├── parameter_server.py   # Parameter Server con sharding y Adagrad
│   ├── model_replica.py      # Model Replica con Downpour SGD
│   └── coordinator.py        # Coordinador principal
├── data/                     # Datos de CIFAR-10 (descargados automáticamente)
├── checkpoints/              # Checkpoints del modelo
└── output/                   # Resultados y métricas
```

## Conceptos Técnicos Implementados

### 1. Parameter Server Sharded

Los parámetros se dividen entre múltiples shards. Si hay 10 shards, cada uno maneja 1/10 de los parámetros:

```python
# Ejemplo con 50,000 parámetros y 2 shards
shard_size = total_params // num_shards  # 25,000 por shard
# Shard 0: parámetros[0:25000]
# Shard 1: parámetros[25000:50000]
```

### 2. Downpour SGD

Algoritmo asíncrono donde cada replica:

1. **Fetch**: Obtiene parámetros actualizados del PS cada `n_fetch` steps
2. **Compute**: Calcula gradientes con un mini-batch
3. **Push**: Envía gradientes al PS cada `n_push` steps

```
while training:
    if step % n_fetch == 0:
        params = async_fetch_from_ps()
    
    loss = compute_gradient(data, params)
    
    if step % n_push == 0:
        async_push_to_ps(gradients)
```

### 3. Adagrad Adaptativo

Learning rate adaptativo por parámetro (implementado localmente en cada shard):

```
η_{i,K} = γ / sqrt(Σ_{j=1}^{K} Δw_{i,j}^2 + ε)
```

### 4. Warm Start

Entrenar primero con **1 sola replica** durante N steps antes de activar las demás. Esto estabiliza el entrenamiento asíncrono.

### 5. Model Parallelism

La red neuronal está diseñada para ser particionada. Cada shard del PS puede manejar diferentes capas del modelo.

## Escalado a Múltiples Máquinas

Para escalar a múltiples máquinas del laboratorio:

### Opción 1: Usar torch.distributed (recomendado)

Modificar `utils.py` para usar `torch.distributed` en vez de multiprocessing local:

```python
# En vez de Manager() y Queue(), usar:
import torch.distributed as dist

dist.init_process_group(backend='gloo')  # Para CPU
# o
dist.init_process_group(backend='nccl')  # Para GPU
```

### Opción 2: Usar MPI

Instalar `mpi4py` y reemplazar la comunicación por MPI:

```bash
pip install mpi4py
```

### Opción 3: TCP/IP directo

Implementar sockets TCP para comunicación entre máquinas:

```python
import socket

# Cada shard escucha en un puerto
# Las replicas se conectan vía TCP
```

### Configuración de Red (para el laboratorio)

1. **Asegurar conectividad**: Todas las máquinas deben verse entre sí
2. **Abrir puertos**: Los shards del PS usan puertos base (default: 29500+)
3. **Compartir filesystem**: Para checkpoints y datos

Ejemplo de lanzamiento en múltiples máquinas:

```bash
# Máquina 1 (Parameter Server + 2 replicas)
python train.py --num-shards 4 --num-replicas 2 --base-port 29500

# Máquina 2 (2 replicas)
python train.py --num-replicas 2 --base-port 29500 --ps-host 192.168.1.10

# Máquina 3 (2 replicas)
python train.py --num-replicas 2 --base-port 29500 --ps-host 192.168.1.10
```

## Troubleshooting

### "RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase"

**Solución**: En Windows, asegúrate de usar `if __name__ == '__main__':`. El script `train.py` ya lo incluye, pero si modificas el código, mantén esta guarda.

### Out of Memory

Reduce el batch size o número de réplicas:
```bash
python train.py --batch-size 32 --num-replicas 2
```

### Lento en CPU

- Usa menos réplicas (cada una es un proceso)
- Reduce el tamaño del modelo
- Usa el modo debug para pruebas

### Errores de comunicación entre procesos

Asegúrate de que no hay procesos zombis:
```bash
# Windows
Task Manager -> Python -> End Task

# Linux
pkill -f train.py
```

## Referencias

1. Dean, J., et al. (2012). "Large Scale Distributed Deep Networks." *NIPS 2012*.
2. Duchi, J., et al. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." *JMLR*.
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Licencia

Este código es para fines educativos y de investigación.
