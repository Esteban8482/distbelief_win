"""
Script principal de entrenamiento para DistBelief en CIFAR-10.

Uso:
    python train.py [opciones]
    
Ejemplos:
    # Entrenamiento básico (usa configuración por defecto)
    python train.py
    
    # Especificar número de réplicas y shards
    python train.py --num-replicas 4 --num-shards 2
    
    # Modo debug (rápido, para probar)
    python train.py --debug
    
    # Sin Adagrad (SGD con lr fijo)
    python train.py --no-adagrad --lr 0.001
    
    # Continuar desde checkpoint
    python train.py --resume ./checkpoints/model_step_1000.pth

Basado en el paper:
"Large Scale Distributed Deep Networks" (Dean et al., NIPS 2012)
"""

import argparse
import sys
import os
import logging
import time

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    DistBeliefConfig, ParameterServerConfig, 
    ModelReplicaConfig, NetworkConfig, DataConfig, TrainingConfig,
    get_default_config, get_debug_config
)
from coordinator import DistBeliefCoordinator


def setup_logging(verbose: bool = False):
    """Configura el logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenamiento distribuido de redes neuronales con DistBelief',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Configuración por defecto (4 réplicas, 2 shards, Adagrad)
  python train.py
  
  # Configuración personalizada
  python train.py --num-replicas 8 --num-shards 4 --batch-size 64
  
  # Modo rápido para pruebas (2 réplicas, 1 shard, menos epochs)
  python train.py --debug
  
  # SGD con learning rate fijo (sin Adagrad)
  python train.py --no-adagrad --lr 0.01
  
  # Ajustar warm start
  python train.py --warm-start-steps 200 --epochs 100
        """
    )
    
    # Configuración del Parameter Server
    ps_group = parser.add_argument_group('Parameter Server')
    ps_group.add_argument(
        '--num-shards', type=int, default=2,
        help='Número de shards del Parameter Server (default: 2)'
    )
    ps_group.add_argument(
        '--no-adagrad', action='store_true',
        help='Usar SGD con learning rate fijo en vez de Adagrad'
    )
    ps_group.add_argument(
        '--adagrad-gamma', type=float, default=0.01,
        help='Gamma (escala global) para Adagrad (default: 0.01)'
    )
    ps_group.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate base (solo si no se usa Adagrad) (default: 0.001)'
    )
    
    # Configuración de Model Replicas
    replica_group = parser.add_argument_group('Model Replicas')
    replica_group.add_argument(
        '--num-replicas', type=int, default=4,
        help='Número de réplicas del modelo (default: 4)'
    )
    replica_group.add_argument(
        '--batch-size', type=int, default=128,
        help='Tamaño del mini-batch (default: 128)'
    )
    replica_group.add_argument(
        '--fetch-freq', type=int, default=1,
        help='Frecuencia de fetch de parámetros n_fetch (default: 1)'
    )
    replica_group.add_argument(
        '--push-freq', type=int, default=1,
        help='Frecuencia de push de gradientes n_push (default: 1)'
    )
    replica_group.add_argument(
        '--warm-start-steps', type=int, default=100,
        help='Steps de warm start antes de activar réplicas (default: 100)'
    )
    
    # Configuración de entrenamiento
    train_group = parser.add_argument_group('Entrenamiento')
    train_group.add_argument(
        '--epochs', type=int, default=50,
        help='Número de epochs (default: 50)'
    )
    train_group.add_argument(
        '--max-steps', type=int, default=0,
        help='Máximo de steps (0 = sin límite) (default: 0)'
    )
    train_group.add_argument(
        '--seed', type=int, default=42,
        help='Semilla aleatoria (default: 42)'
    )
    
    # Directorios
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
    
    # Otros
    parser.add_argument(
        '--debug', action='store_true',
        help='Modo debug: configuración reducida para pruebas rápidas'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Logging verboso'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Solo evaluar (requiere --resume)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Continuar desde checkpoint'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> DistBeliefConfig:
    """
    Crea la configuración a partir de los argumentos.
    
    Args:
        args: Argumentos parseados
        
    Returns:
        DistBeliefConfig
    """
    if args.debug:
        config = get_debug_config()
        print("=" * 60)
        print("MODO DEBUG ACTIVADO")
        print("=" * 60)
    else:
        config = get_default_config()
    
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
    config.training.resume_from = args.resume
    config.training.output_dir = args.output_dir
    
    # Datos
    config.data.data_dir = args.data_dir
    
    return config


def print_config(config: DistBeliefConfig):
    """Imprime la configuración de forma legible."""
    print("\n" + "=" * 60)
    print("CONFIGURACION DEL SISTEMA")
    print("=" * 60)
    
    print("\n[Parameter Server]")
    print(f"  Shards:              {config.ps.num_shards}")
    print(f"  Adagrad:             {'Si' if config.ps.use_adagrad else 'No (SGD)'}")
    if config.ps.use_adagrad:
        print(f"  Adagard gamma:       {config.ps.adagrad_gamma}")
    else:
        print(f"  Learning rate:       {config.ps.base_learning_rate}")
    print(f"  Checkpoint dir:      {config.ps.checkpoint_dir}")
    
    print("\n[Model Replicas]")
    print(f"  Numero de replicas:  {config.replica.num_replicas}")
    print(f"  Batch size:          {config.replica.batch_size}")
    print(f"  Fetch frequency:     {config.replica.fetch_frequency}")
    print(f"  Push frequency:      {config.replica.push_frequency}")
    print(f"  Warm start steps:    {config.replica.warm_start_steps}")
    
    print("\n[Entrenamiento]")
    print(f"  Epochs:              {config.training.num_epochs}")
    print(f"  Max steps:           {config.training.max_steps or 'Sin limite'}")
    print(f"  Seed:                {config.training.seed}")
    
    print("\n[Datos]")
    print(f"  Dataset:             CIFAR-10")
    print(f"  Directorio:          {config.data.data_dir}")
    
    print("\n[Output]")
    print(f"  Directorio:          {config.training.output_dir}")
    print("=" * 60 + "\n")


def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("  DistBelief - Distributed Deep Learning")
    print("  Basado en Dean et al. (NIPS 2012)")
    print("=" * 60)
    
    # Crear configuración
    config = create_config_from_args(args)
    
    # Mostrar configuración
    print_config(config)
    
    # Detectar sistema operativo
    import platform
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPUs disponibles: {os.cpu_count()}")
    print()
    
    # Verificar requisitos
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"Dispositivo: {'CUDA disponible' if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("ERROR: PyTorch no esta instalado.")
        print("Instalar con: pip install torch torchvision")
        return 1
    
    try:
        import torchvision
        print(f"Torchvision: {torchvision.__version__}")
    except ImportError:
        print("ERROR: torchvision no esta instalado.")
        print("Instalar con: pip install torchvision")
        return 1
    
    print()
    
    # Confirmar inicio
    if not args.debug:
        print("El entrenamiento comenzara en 3 segundos...")
        print("Presiona Ctrl+C para cancelar\n")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nCancelado por el usuario.")
            return 0
    
    # Ejecutar entrenamiento
    print("\nIniciando entrenamiento...\n")
    
    coordinator = DistBeliefCoordinator(config)
    results = coordinator.run_full_training()
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    
    if results['success']:
        eval_results = results.get('eval_results', {})
        print(f"Test Loss:      {eval_results.get('test_loss', 'N/A')}")
        print(f"Test Accuracy:  {eval_results.get('test_accuracy', 'N/A')}%")
        print(f"Correctos:      {eval_results.get('correct', 'N/A')}/{eval_results.get('total', 'N/A')}")
        print("\nEntrenamiento completado exitosamente!")
    else:
        print(f"El entrenamiento no completo: {results.get('reason', 'Error desconocido')}")
        return 1
    
    return 0


if __name__ == '__main__':
    # En Windows, multiprocessing requiere esta guarda
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    sys.exit(main())
