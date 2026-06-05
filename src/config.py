"""
Configuración del sistema DistBelief para entrenamiento en CIFAR-10.

Basado en el paper "Large Scale Distributed Deep Networks" (Dean et al., NIPS 2012)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import os


@dataclass
class ParameterServerConfig:
    """Configuración del Parameter Server sharded.
    
    Según el paper (Sección 4.1): "sharded across many machines 
    (e.g., if we have 10 parameter server shards, each shard is 
    responsible for storing and applying updates to 1/10th of the 
    model parameters)"
    """
    num_shards: int = 2
    base_port: int = 29500
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 500
    adagrad_gamma: float = 0.01
    adagrad_epsilon: float = 1e-10
    use_adagrad: bool = True
    base_learning_rate: float = 0.001


@dataclass
class ModelReplicaConfig:
    """Configuración de cada Model Replica.
    
    Según el paper (Sección 4.1): "We divide the training data into 
    a number of subsets and run a copy of the model on each of these 
    subsets."
    """
    num_replicas: int = 4
    batch_size: int = 128
    fetch_frequency: int = 1
    push_frequency: int = 1
    warm_start_steps: int = 100
    device: str = "cpu"
    log_frequency: int = 10
    gradient_clipping: bool = True
    max_gradient_norm: float = 5.0


@dataclass
class NetworkConfig:
    """Configuración de la red neuronal para CIFAR-10."""
    input_channels: int = 3
    input_height: int = 32
    input_width: int = 32
    num_classes: int = 10
    conv_layers: List[tuple] = field(default_factory=lambda: [
        (64, 3, 1, 1),   # conv1
        (128, 3, 1, 1),  # conv2  
        (256, 3, 1, 1),  # conv3
    ])
    pool_size: int = 2
    dropout_rate: float = 0.5
    fc_layers: List[int] = field(default_factory=lambda: [512, 256])
    use_batch_norm: bool = True


@dataclass
class DataConfig:
    """Configuración del dataset CIFAR-10."""
    data_dir: str = "./data"
    num_workers: int = 2
    validation_split: float = 0.1
    random_crop: int = 32
    random_horizontal_flip: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalize_std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass
class TrainingConfig:
    """Configuración del entrenamiento."""
    num_epochs: int = 50
    max_steps: int = 0
    eval_frequency: int = 200
    save_frequency: int = 500
    seed: int = 42
    resume_from: Optional[str] = None
    output_dir: str = "./output"
    use_amp: bool = False


@dataclass
class DistBeliefConfig:
    """Configuración completa del sistema DistBelief."""
    ps: ParameterServerConfig = field(default_factory=ParameterServerConfig)
    replica: ModelReplicaConfig = field(default_factory=ModelReplicaConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def validate(self):
        """Valida la configuración."""
        assert self.ps.num_shards > 0, "Debe haber al menos 1 shard"
        assert self.replica.num_replicas > 0, "Debe haber al menos 1 replica"
        assert self.replica.batch_size > 0, "Batch size debe ser positivo"
        
        # Crear directorios necesarios
        os.makedirs(self.ps.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.training.output_dir, exist_ok=True)


def get_default_config() -> DistBeliefConfig:
    """Retorna la configuración por defecto optimizada para CIFAR-10."""
    return DistBeliefConfig()


def get_debug_config() -> DistBeliefConfig:
    """Configuración reducida para debug rápido."""
    config = DistBeliefConfig()
    config.ps.num_shards = 1
    config.replica.num_replicas = 2
    config.replica.batch_size = 32
    config.replica.warm_start_steps = 20
    config.training.num_epochs = 2
    config.training.eval_frequency = 50
    return config
