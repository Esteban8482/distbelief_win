# DistBelief - Implementación de Distributed Deep Learning

Implementación del framework **DistBelief** descrito en el paper
[*"Large Scale Distributed Deep Networks"*](https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html)
de Dean et al. (NIPS 2012), con entrenamiento en **CIFAR-10**.

Soporta entrenamiento **local** (una máquina), **distribuido** (múltiples
máquinas), y **PS sharded distribuido** (cada shard del Parameter Server
en una máquina diferente).

## Arquitectura del Sistema

### Modo Local
```
+-------------------------------+
|           [TU PC]             |
|  +---------+ +-------------+  |
|  | PS (1)  | | Replicas(4) |  |
|  | -Queue  | | -Queue      |  |
|  +---------+ +-------------+  |
+-------------------------------+
```

### Modo Distribuido (PS en una máquina)
```
+---------------+        +---------------------+
| [PC A - PS]   |<TCP>   | [PC B - Replicas]   |
| :29500        |        | 4 replicas          |
+---------------+        +---------------------+
      ^
      |<TCP>
      v
+---------------------+
| [PC C - Replicas]   |
| 8 replicas          |
+---------------------+
```

### PS Sharded Distribuido (cada shard en una máquina)
```
+---------------+        +---------------+        +---------------------+
| [PC A]        |        | [PC B]        |        | [PC C - Replicas]   |
| PS Shard 0    |<TCP>   | PS Shard 1    |<TCP>   | 4 replicas          |
| :29500        |        | :29500        |        |                     |
+---------------+        +---------------+        +---------------------+
      ^                        ^
      |<TCP>                   |<TCP>
      v                        v
+---------------------+
| [PC D - Replicas]   |
| 8 replicas          |
+---------------------+
```

## Características Implementadas

| Característica | Descripción | Paper Ref. |
|---|---|---|
| **Parameter Server Sharded** | Parámetros divididos entre shards | Sección 4.1 |
| **Downpour SGD** | SGD asíncrono con fetch/push | Sección 4.1 |
| **Adagrad Adaptativo** | Learning rate adaptativo por parámetro | Sección 4.1 |
| **Model Replicas** | Múltiples réplicas en paralelo | Sección 4.1 |
| **Warm Start** | 1 replica primero, luego las demás | Sección 4.1 |
| **Modo Distribuido** | PS y réplicas en máquinas distintas | - |
| **PS Sharded Distribuido** | Cada shard del PS en una máquina diferente | - |
| **Transporte TCP/IP** | Comunicación entre máquinas vía sockets | - |
| **Gradient Clipping** | Clip de gradientes | - |
| **Logging a archivo** | Auditoría persistente | - |

## Instalación

```bash
pip install -r requirements.txt
```

Requisitos: Python 3.7+, PyTorch 1.9+, Windows 10/Linux/macOS.

## Uso

### Modo Local (todo en una máquina)

```bash
# Configuración por defecto (4 réplicas, 2 shards, Adagrad)
python train.py

# 8 réplicas, 4 shards, batch 64
python train.py --num-replicas 8 --num-shards 4 --batch-size 64

# Debug rápido (2 réplicas, 1 shard, 2 epochs)
python train.py --debug
```

### Modo Distribuido (PS en una máquina)

**Máquina A** (Parameter Server):
```bash
python train.py --role ps --num-shards 1 --ps-port-base 29500
```

**Máquina B** (4 réplicas):
```bash
python train.py --role replica --ps-host 192.168.1.10 --ps-port-base 29500 \
                --num-replicas 4 --replica-offset 0
```

**Máquina C** (8 réplicas):
```bash
python train.py --role replica --ps-host 192.168.1.10 --ps-port-base 29500 \
                --num-replicas 8 --replica-offset 4
```

### PS Sharded Distribuido (cada shard en una máquina)

En este modo, **cada shard del PS corre en una máquina diferente**.
Las réplicas se conectan a todos los shards vía `--shard-endpoints`.

**Máquina A** (PS Shard 0):
```bash
python train.py --role ps --shard-id 0 --num-shards 2 --ps-port-base 29500
```

**Máquina B** (PS Shard 1):
```bash
python train.py --role ps --shard-id 1 --num-shards 2 --ps-port-base 29500
```

**Máquina C** (4 réplicas):
```bash
python train.py --role replica \
    --shard-endpoints 192.168.1.10:29500,192.168.1.11:29500 \
    --num-replicas 4 --batch-size 128
```

**Máquina D** (8 réplicas):
```bash
python train.py --role replica \
    --shard-endpoints 192.168.1.10:29500,192.168.1.11:29500 \
    --num-replicas 8 --replica-offset 4 --batch-size 128
```

### Configuración de red

**Abrir puertos en firewall (Windows)**:
```powershell
New-NetFirewallRule -DisplayName "DistBelief" -Direction Inbound -Protocol TCP -LocalPort 29500-29510 -Action Allow
```

**Verificar conectividad**:
```bash
# Desde máquina réplica hacia PS
python -c "import socket; s=socket.socket(); s.connect(('192.168.1.10', 29500)); print('OK')"
```

### Usando archivo de configuración del cluster

Copia `cluster_config_example.json` como `cluster.json` y ajusta las IPs.

**Para PS sharded distribuido**:
```json
{
  "parameter_server": {
    "num_shards": 2,
    "shard_endpoints": ["192.168.1.10:29500", "192.168.1.11:29500"]
  },
  "nodes": [
    {"node_id": 0, "role": "ps", "num_replicas": 0},
    {"node_id": 1, "role": "ps", "num_replicas": 0},
    {"node_id": 2, "role": "replica", "num_replicas": 4, "replica_offset": 0},
    {"node_id": 3, "role": "replica", "num_replicas": 8, "replica_offset": 4}
  ]
}
```

Lanzar:
```bash
# Máquina A (PS Shard 0)
python train.py --cluster-config cluster.json --node-id 0

# Máquina B (PS Shard 1)
python train.py --cluster-config cluster.json --node-id 1

# Máquina C (4 réplicas)
python train.py --cluster-config cluster.json --node-id 2

# Máquina D (8 réplicas)
python train.py --cluster-config cluster.json --node-id 3
```

### Opciones completas

```
Rol del Nodo:
  --role {coordinator,ps,replica}   Rol de este nodo
  --shard-id N                      ID del shard a lanzar (con --role ps)
  --ps-host HOST                    IP del PS
  --ps-port-base PORT               Puerto base del PS (default: 29500)
  --shard-endpoints "h:p,h:p"       Endpoints de cada shard (PS sharded distribuido)
  --replica-offset N                Offset de IDs de réplicas
  --cluster-config FILE             Archivo JSON de configuración del cluster
  --node-id ID                      ID de nodo en el cluster

Parameter Server:
  --num-shards N                    Shards del PS (default: 2)
  --no-adagrad                      Usar SGD en vez de Adagrad
  --adagrad-gamma F                 Gamma de Adagrad (default: 0.01)
  --lr F                            Learning rate base (default: 0.001)

Model Replicas:
  --num-replicas N                  Réplicas en este nodo (default: 4)
  --batch-size N                    Batch size (default: 128)
  --fetch-freq N                    Frecuencia fetch n_fetch (default: 1)
  --push-freq N                     Frecuencia push n_push (default: 1)
  --warm-start-steps N              Warm start (default: 100)

Entrenamiento:
  --epochs N                        Epochs (default: 50)
  --max-steps N                     Máximo steps (0=sin límite)
  --seed N                          Semilla aleatoria (default: 42)

Rutas:
  --data-dir DIR                    Directorio CIFAR-10 (default: ./data)
  --output-dir DIR                  Directorio salida (default: ./output)
  --checkpoint-dir DIR              Directorio checkpoints (default: ./checkpoints)

  --debug                           Modo debug rápido
  --verbose, -v                     Logging verboso
```

## Estructura del Proyecto

```
distbelief/
├── train.py                          # Script principal
├── requirements.txt                  # Dependencias
├── README.md                         # Este archivo
├── setup.bat                         # Setup para Windows
├── train_quick.bat                   # Entrenamiento rápido local
├── launch_cluster.bat                # Lanzador de cluster
├── cluster_config_example.json       # Ejemplo de config de cluster
├── data/                             # CIFAR-10 (auto-descargado)
├── checkpoints/                      # Checkpoints del modelo
├── output/                           # Resultados y métricas
└── src/
    ├── __init__.py
    ├── config.py                     # Configuración
    ├── utils.py                      # Utilidades
    ├── networking.py                 # Capa de transporte TCP/IP
    ├── network.py                    # Red neuronal CIFAR10Net
    ├── parameter_server.py           # PS con sharding y Adagrad
    ├── model_replica.py              # Réplica con Downpour SGD
    ├── coordinator.py                # Coordinador (local + distribuido)
    └── communication_hub.py          # Hub de comunicación legacy
```

## Protocolo de Comunicación TCP

El protocolo entre PS y réplicas usa **pickle sobre sockets TCP**:

```
[4 bytes: tamaño del payload][N bytes: payload pickle(Message)]
```

**Handshake** (réplica → PS al conectar):
```python
Message(msg_type=START_TRAINING, data={'replica_id': id})
```

**Fetch de parámetros** (asíncrono):
```python
# Réplica → PS Shard N:
Message(msg_type=GET_PARAMETERS, sender_id=id, data={})

# PS → Réplica:
Message(msg_type=PARAMETERS_RESPONSE, data={
    'shard_id': N, 'start_idx': 0, 'end_idx': M,
    'parameters': torch.Tensor([...])  # ~10 MB para CIFAR10Net
})
```

**Push de gradientes** (asíncrono):
```python
# Réplica → PS Shard N:
Message(msg_type=PUSH_GRADIENTS, data={
    'shard_id': N, 'start_idx': 0, 'end_idx': M,
    'gradients': torch.Tensor([...])
})

# PS → Réplica:
Message(msg_type=GRADIENTS_ACK, data={'status': 'applied'})
```

## Troubleshooting

### "Connection refused" al conectar réplicas al PS

Verificar:
1. El PS está corriendo y escuchando: `netstat -an | findstr 29500`
2. Las IPs son accesibles: `ping 192.168.1.10`
3. Los puertos están abiertos en el firewall
4. No hay otro proceso usando el puerto

### Réplica se queda esperando warm start indefinidamente

- Verificar que TODOS los shards del PS están corriendo
- Verificar conectividad a CADA shard: `--shard-endpoints`
- Revisar log: `output/logs/distbelief_training.log`

### OOM (Out of Memory)

- Reducir `--batch-size` (probar 32, 64)
- Reducir `--num-replicas`
- Verificar procesos zombis: `tasklist | findstr python`

## Referencias

1. Dean, J., et al. (2012). "Large Scale Distributed Deep Networks." *NIPS 2012*.
2. Duchi, J., et al. (2011). "Adaptive Subgradient Methods." *JMLR*.
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
