# DistBelief - Implementación de Distributed Deep Learning

Implementación del framework **DistBelief** descrito en el paper [*"Large Scale Distributed Deep Networks"*](https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html) de Dean et al. (NIPS 2012), con entrenamiento en **CIFAR-10**.

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
- **RAM**: 4GB mínimo
- **CPU**: Cuantos más cores, mejor ya que cada replica usa 1 proceso

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

## Referencias

1. Dean, J., et al. (2012). "Large Scale Distributed Deep Networks." *NIPS 2012*.