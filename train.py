"""
Script principal de entrenamiento para DistBelief en CIFAR-10.

Soporta tres modos de operación:
  1. LOCAL: Todo en una máquina (multiprocessing.Queue)
  2. DISTRIBUIDO-PS: Solo Parameter Server (escucha en TCP)
  3. DISTRIBUIDO-REPLICA: Solo Model Replicas (conectan vía TCP al PS)

Uso:
  # Modo local (una máquina)
  python train.py --num-replicas 4 --num-shards 1

  # Modo distribuido - Máquina 1 (PS):
  python train.py --role ps --num-shards 1 --ps-port-base 29500

  # Modo distribuido - Máquina 2 (4 réplicas):
  python train.py --role replica --ps-host 192.168.1.10 --ps-port-base 29500 \\
                  --num-replicas 4 --replica-offset 0

  # Modo distribuido - Máquina 3 (8 réplicas):
  python train.py --role replica --ps-host 192.168.1.10 --ps-port-base 29500 \\
                  --num-replicas 8 --replica-offset 4

  # Desde archivo de configuración de cluster:
  python train.py --cluster-config cluster.json --role ps
  python train.py --cluster-config cluster.json --role replica --node-id 1
"""

import argparse
import sys
import os
import logging
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    DistBeliefConfig, ParameterServerConfig,
    ModelReplicaConfig, NetworkConfig, DataConfig, TrainingConfig,
    get_default_config, get_debug_config
)
from coordinator import DistBeliefCoordinator


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='DistBelief - Distributed Deep Learning (Dean et al., NIPS 2012)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # LOCAL - Todo en una maquina
  python train.py --num-replicas 4 --num-shards 1 --batch-size 128

  # DIST - PS en una maquina, replicas en otras:
  #   Maquina A (PS):  python train.py --role ps --num-shards 1 --ps-port-base 29500
  #   Maquina B (4 replicas): python train.py --role replica --ps-host 192.168.1.10 --num-replicas 4

  # DIST - PS SHARDED (2 maquinas como PS, replicas en otras):
  #   Maquina A (PS Shard 0): python train.py --role ps --shard-id 0 --num-shards 2 --ps-port-base 29500
  #   Maquina B (PS Shard 1): python train.py --role ps --shard-id 1 --num-shards 2 --ps-port-base 29500
  #   Maquina C (4 replicas): python train.py --role replica --shard-endpoints 192.168.1.10:29500,192.168.1.11:29500 --num-replicas 4

  # DEBUG rapido
  python train.py --debug
        """
    )

    # --- Rol del nodo (clave para modo distribuido) ---
    role_group = parser.add_argument_group('Rol del Nodo')
    role_group.add_argument(
        '--role', type=str, default='coordinator',
        choices=['coordinator', 'ps', 'replica'],
        help='Rol de este nodo: coordinator (todo local), ps (solo PS), replica (solo réplicas)'
    )
    role_group.add_argument(
        '--shard-id', type=int, default=None,
        help='ID del shard a lanzar (solo con --role ps; si no se especifica, lanza todos)'
    )
    role_group.add_argument(
        '--ps-host', type=str, default=None,
        help='IP/hostname del Parameter Server (requerido si role=replica y no se usa --shard-endpoints)'
    )
    role_group.add_argument(
        '--ps-port-base', type=int, default=29500,
        help='Puerto base del PS (default: 29500). En PS sharded distribuido, cada shard usa ps_port_base + shard_id'
    )
    role_group.add_argument(
        '--shard-endpoints', type=str, default=None,
        help='Endpoints de cada shard: "host1:port1,host2:port2,...". Sobrescribe --ps-host y --ps-port-base.'
    )
    role_group.add_argument(
        '--replica-offset', type=int, default=0,
        help='Offset de IDs para réplicas en este nodo (para múltiples nodos réplica)'
    )
    role_group.add_argument(
        '--cluster-config', type=str, default=None,
        help='Archivo JSON con configuración del cluster'
    )
    role_group.add_argument(
        '--node-id', type=int, default=None,
        help='ID de este nodo en el cluster (requiere --cluster-config)'
    )

    # --- Parameter Server ---
    ps_group = parser.add_argument_group('Parameter Server')
    ps_group.add_argument(
        '--num-shards', type=int, default=2,
        help='Número de shards del PS (default: 2)'
    )
    ps_group.add_argument(
        '--no-adagrad', action='store_true',
        help='Usar SGD con lr fijo en vez de Adagrad'
    )
    ps_group.add_argument(
        '--adagrad-gamma', type=float, default=0.01,
        help='Gamma para Adagrad (default: 0.01)'
    )
    ps_group.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate base sin Adagrad (default: 0.001)'
    )

    # --- Model Replicas ---
    replica_group = parser.add_argument_group('Model Replicas')
    replica_group.add_argument(
        '--num-replicas', type=int, default=4,
        help='Número de réplicas en ESTE nodo (default: 4)'
    )
    replica_group.add_argument(
        '--batch-size', type=int, default=128,
        help='Tamaño del mini-batch (default: 128)'
    )
    replica_group.add_argument(
        '--fetch-freq', type=int, default=1,
        help='Frecuencia de fetch n_fetch (default: 1)'
    )
    replica_group.add_argument(
        '--push-freq', type=int, default=1,
        help='Frecuencia de push n_push (default: 1)'
    )
    replica_group.add_argument(
        '--warm-start-steps', type=int, default=100,
        help='Steps de warm start (default: 100)'
    )

    # --- Entrenamiento ---
    train_group = parser.add_argument_group('Entrenamiento')
    train_group.add_argument(
        '--epochs', type=int, default=50,
        help='Número de epochs (default: 50)'
    )
    train_group.add_argument(
        '--max-steps', type=int, default=0,
        help='Máximo de steps (0=sin límite) (default: 0)'
    )
    train_group.add_argument(
        '--seed', type=int, default=42,
        help='Semilla aleatoria (default: 42)'
    )

    # --- Directorios ---
    path_group = parser.add_argument_group('Rutas')
    path_group.add_argument(
        '--data-dir', type=str, default='./data',
        help='Directorio para CIFAR-10 (default: ./data)'
    )
    path_group.add_argument(
        '--output-dir', type=str, default='./output',
        help='Directorio de salida (default: ./output)'
    )
    path_group.add_argument(
        '--checkpoint-dir', type=str, default='./checkpoints',
        help='Directorio para checkpoints (default: ./checkpoints)'
    )

    # --- Otros ---
    parser.add_argument(
        '--debug', action='store_true',
        help='Modo debug: configuración reducida'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Logging verboso'
    )

    return parser.parse_args()


def load_cluster_config(filepath: str, node_id: int) -> dict:
    """Carga configuración de nodo desde archivo de cluster."""
    with open(filepath, 'r') as f:
        config = json.load(f)

    node_config = config['nodes'][node_id]
    ps_config = config['parameter_server']

    return {
        'role': node_config['role'],
        'ps_host': ps_config['host'],
        'ps_port_base': ps_config['port_base'],
        'num_shards': ps_config['num_shards'],
        'num_replicas': node_config.get('num_replicas', 0),
        'replica_offset': node_config.get('replica_offset', 0),
    }


def create_config_from_args(args) -> tuple:
    """Crea configuración a partir de argumentos."""
    if args.debug:
        config = get_debug_config()
        print("=" * 60)
        print("MODO DEBUG ACTIVADO")
        print("=" * 60)
    else:
        config = get_default_config()

    # Si se especifica cluster-config, cargar desde archivo
    role = args.role
    ps_host = args.ps_host
    ps_port_base = args.ps_port_base
    replica_offset = args.replica_offset
    shard_endpoints = None

    # Parsear --shard-endpoints si se proporciona (sobrescribe ps_host/ps_port_base)
    if args.shard_endpoints:
        shard_endpoints = []
        for ep in args.shard_endpoints.split(','):
            host, port = ep.strip().split(':')
            shard_endpoints.append((host, int(port)))
        config.ps.num_shards = len(shard_endpoints)

    if args.cluster_config and args.node_id is not None:
        cluster = load_cluster_config(args.cluster_config, args.node_id)
        role = cluster['role']
        ps_host = cluster['ps_host']
        ps_port_base = cluster['ps_port_base']
        config.ps.num_shards = cluster['num_shards']
        config.replica.num_replicas = cluster['num_replicas']
        replica_offset = cluster['replica_offset']

    # Parameter Server
    config.ps.num_shards = args.num_shards
    config.ps.use_adagrad = not args.no_adagrad
    config.ps.adagrad_gamma = args.adagrad_gamma
    config.ps.base_learning_rate = args.lr
    config.ps.checkpoint_dir = args.checkpoint_dir

    # Model Replicas
    config.replica.num_replicas = args.num_replicas
    config.replica.batch_size = args.batch_size
    config.replica.fetch_frequency = args.fetch_freq
    config.replica.push_frequency = args.push_freq
    config.replica.warm_start_steps = args.warm_start_steps

    # Entrenamiento
    config.training.num_epochs = args.epochs
    config.training.max_steps = args.max_steps
    config.training.seed = args.seed
    config.training.output_dir = args.output_dir

    # Datos
    config.data.data_dir = args.data_dir

    shard_id = args.shard_id

    return config, role, ps_host, ps_port_base, replica_offset, shard_id, shard_endpoints


def print_config(config: DistBeliefConfig, role: str, ps_host, ps_port_base):
    """Imprime configuración."""
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DEL SISTEMA")
    print("=" * 60)

    print(f"\n[Nodo]")
    print(f"  Rol:              {role}")
    if role == 'replica' and ps_host:
        print(f"  PS remoto:        {ps_host}:{ps_port_base}")

    print(f"\n[Parameter Server]")
    print(f"  Shards:           {config.ps.num_shards}")
    print(f"  Adagrad:          {'Sí' if config.ps.use_adagrad else 'No (SGD)'}")
    if config.ps.use_adagrad:
        print(f"  Adagard gamma:    {config.ps.adagrad_gamma}")

    print(f"\n[Model Replicas]")
    print(f"  Réplicas:         {config.replica.num_replicas}")
    print(f"  Batch size:       {config.replica.batch_size}")

    print(f"\n[Entrenamiento]")
    print(f"  Epochs:           {config.training.num_epochs}")
    print(f"  Warm start:       {config.replica.warm_start_steps} steps")

    print("=" * 60 + "\n")


def main():
    """Función principal."""
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H-%m-%d %H:%M:%S'
    )

    print("\n" + "=" * 60)
    print("  DistBelief - Distributed Deep Learning")
    print("  Dean et al. (NIPS 2012)")
    print("=" * 60)

    # Validar argumentos
    if args.role == 'replica' and not args.ps_host and not args.cluster_config:
        print("\nERROR: --ps-host es requerido cuando --role=replica")
        print("Ejemplo: python train.py --role replica --ps-host 192.168.1.10")
        return 1

    config, role, ps_host, ps_port_base, replica_offset, shard_id, shard_endpoints = create_config_from_args(args)
    print_config(config, role, ps_host, ps_port_base)

    # Info del sistema
    import platform
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPUs: {os.cpu_count()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {'Sí' if torch.cuda.is_available() else 'No'}")
    except ImportError:
        print("ERROR: PyTorch no instalado.")
        return 1

    print()

    if not args.debug and role in ('coordinator', 'replica'):
        print("El entrenamiento comenzará en 3 segundos...")
        print("Presiona Ctrl+C para cancelar\n")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nCancelado.")
            return 0

    # Crear y ejecutar coordinador
    coordinator = DistBeliefCoordinator(
        config=config,
        role=role,
        ps_host=ps_host,
        ps_port_base=ps_port_base,
        shard_id=shard_id,
        shard_endpoints=shard_endpoints
    )

    results = coordinator.run_full_training()

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    if results['success']:
        eval_results = results.get('eval_results', {})
        print(f"Test Loss:     {eval_results.get('test_loss', 'N/A')}")
        print(f"Test Accuracy: {eval_results.get('test_accuracy', 'N/A')}%")
        print("\nEntrenamiento completado!")
    else:
        print(f"No completó: {results.get('reason', 'Error')}")
        return 1

    return 0


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
