"""
Model Replica con Downpour SGD asíncrono.

Soporta dos modos de conexión al Parameter Server:
- LOCAL: multiprocessing.Queue (todo en una máquina)
- DISTRIBUIDO: TCP/IP (se conecta al PS en otra máquina)

Downpour SGD puro: la réplica solo fetchea parámetros, computa gradientes,
y envía al PS. El PS (con Adagrad) es la única fuente de verdad.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Any, Union
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time
import logging
import os

from utils import (
    Message, MessageType,
    get_parameter_shard_bounds, clip_gradients, split_gradients_by_shard
)
from config import ModelReplicaConfig, ParameterServerConfig
from network import CIFAR10Net, count_parameters

# Import condicional de networking (solo en modo distribuido)
try:
    from networking import ModelReplicaClient, _send_message, _recv_message
    NETWORKING_AVAILABLE = True
except ImportError:
    NETWORKING_AVAILABLE = False


logger = logging.getLogger("ModelReplica")


class ReplicaTransport:
    """
    Abstracción de transporte para la réplica.
    
    Unifica la interfaz entre modo LOCAL (Queue) y DISTRIBUIDO (TCP),
    de forma que el resto del código no necesita saber qué modo se usa.
    """
    
    def __init__(self, replica_id: int, mode: str = "local"):
        self.replica_id = replica_id
        self.mode = mode  # "local" o "distributed"
        
        # Modo local
        self.local_request_queues: Optional[List[Queue]] = None
        self.local_response_queue: Optional[Queue] = None
        
        # Modo distribuido
        self.tcp_client: Optional[Any] = None
        self.shard_endpoints: Optional[List[tuple]] = None
    
    def setup_local(self, request_queues: List[Queue], response_queue: Queue):
        """Configura transporte local con Queue."""
        self.mode = "local"
        self.local_request_queues = request_queues
        self.local_response_queue = response_queue
    
    def setup_distributed(self, shard_endpoints: List[tuple]) -> bool:
        """Configura transporte distribuido con TCP."""
        if not NETWORKING_AVAILABLE:
            logger.error("Módulo networking no disponible. No se puede usar modo distribuido.")
            return False
        
        self.mode = "distributed"
        self.shard_endpoints = shard_endpoints
        self.tcp_client = ModelReplicaClient(self.replica_id, shard_endpoints)
        return self.tcp_client.connect()
    
    def send_to_shard(self, shard_id: int, msg: Message):
        """Envía un mensaje a un shard del PS."""
        if self.mode == "local":
            if self.local_request_queues and shard_id < len(self.local_request_queues):
                try:
                    self.local_request_queues[shard_id].put(msg, block=False)
                except queue.Full:
                    pass
        elif self.mode == "distributed":
            if self.tcp_client:
                self.tcp_client.send_to_shard(shard_id, msg)
    
    def receive(self, block: bool = True, timeout: float = 1.0) -> Optional[Message]:
        """Recibe un mensaje del PS."""
        if self.mode == "local":
            if self.local_response_queue:
                try:
                    return self.local_response_queue.get(block=block, timeout=timeout)
                except queue.Empty:
                    return None
        elif self.mode == "distributed":
            if self.tcp_client:
                return self.tcp_client.receive(block=block, timeout=timeout)
        return None
    
    def num_shards(self) -> int:
        """Retorna el número de shards disponibles."""
        if self.mode == "local" and self.local_request_queues:
            return len(self.local_request_queues)
        elif self.mode == "distributed" and self.tcp_client:
            return len(self.tcp_client.socks)
        return 0
    
    def cleanup(self):
        """Limpia recursos de transporte."""
        if self.mode == "distributed" and self.tcp_client:
            self.tcp_client.disconnect()


def model_replica_process(
    replica_id: int,
    ps_request_queues: Optional[List[Queue]],
    ps_response_queue: Optional[Queue],
    should_stop,
    warm_start_done,
    global_step,
    replica_config_dict: dict,
    ps_config_dict: dict,
    total_params: int,
    train_data_indices: List[int],
    data_dir: str = "./data",
    is_warm_start: bool = False,
    shard_endpoints: Optional[List[tuple]] = None
):
    """
    Proceso para una Model Replica con Downpour SGD.
    
    Args:
        replica_id: ID único de la réplica
        ps_request_queues: Colas de requests (modo local) o None
        ps_response_queue: Cola de responses (modo local) or None
        should_stop: Value compartido para señal de parada
        warm_start_done: Value compartido
        global_step: Value compartido con el step global
        replica_config_dict: Configuración de la replica como dict
        ps_config_dict: Configuración del PS como dict
        total_params: Número total de parámetros
        train_data_indices: Índices del dataset
        data_dir: Directorio de datos
        is_warm_start: Si es la replica de warm start
        shard_endpoints: Lista de (host, port) para shards (modo distribuido)
    """
    # Recrear configuraciones
    replica_config = ModelReplicaConfig(**replica_config_dict)
    ps_config = ParameterServerConfig(**ps_config_dict)
    
    # Configurar logging
    _setup_replica_logging(replica_id)
    
    logger.info(f"Iniciando Model Replica {replica_id} (warm_start={is_warm_start})")
    
    # Configurar transporte (LOCAL o DISTRIBUIDO)
    transport = ReplicaTransport(replica_id)
    
    if shard_endpoints is not None:
        # MODO DISTRIBUIDO: conectar vía TCP al PS
        logger.info(f"Réplica {replica_id}: Modo DISTRIBUIDO, conectando a {shard_endpoints}")
        if not transport.setup_distributed(shard_endpoints):
            logger.error(f"Réplica {replica_id}: Fallo conectando al PS. Abortando.")
            return
    else:
        # MODO LOCAL: usar Queue
        logger.info(f"Réplica {replica_id}: Modo LOCAL, usando Queue")
        transport.setup_local(ps_request_queues, ps_response_queue)
    
    num_shards = ps_config.num_shards
    
    # Configurar device
    device = torch.device(replica_config.device)
    
    # Crear modelo
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Fetch inicial de parámetros
    logger.info(f"Replica {replica_id}: Fetch inicial de parámetros")
    _fetch_all_parameters(
        replica_id, transport, model, total_params, num_shards, device
    )
    
    # Preparar dataset
    from torchvision import datasets, transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])
    
    full_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transform_train
    )
    
    if len(train_data_indices) < len(full_dataset):
        replica_dataset = Subset(full_dataset, train_data_indices)
    else:
        replica_dataset = full_dataset
    
    dataloader = DataLoader(
        replica_dataset,
        batch_size=replica_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Loop de entrenamiento
    logger.info(f"Replica {replica_id}: Comenzando entrenamiento")
    
    step = 0
    accrued_gradients = None
    warm_start_already_signaled = False
    
    try:
        epoch = 0
        max_epochs = 50
        
        while not should_stop.value and epoch < max_epochs:
            for batch_idx, (data, target) in enumerate(dataloader):
                if should_stop.value:
                    break
                
                # Verificar warm start
                if not is_warm_start and not warm_start_done.value:
                    time.sleep(0.5)
                    continue
                
                data, target = data.to(device), target.to(device)
                
                # 1. Fetch parámetros cada n_fetch steps
                if step % replica_config.fetch_frequency == 0:
                    _async_fetch_parameters(replica_id, transport, num_shards)
                
                # Consumir parámetros del PS (descarta ACKs)
                _consume_parameters_from_ps(transport, model, total_params, device)
                
                # 2. Forward + backward (computar gradientes)
                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Calcular accuracy
                with torch.no_grad():
                    _, predicted = output.max(1)
                    correct = predicted.eq(target).sum().item()
                    total = target.size(0)
                    batch_accuracy = 100.0 * correct / total
                
                # 3. Obtener gradientes y acumular
                gradients = model.get_flat_gradients()
                
                if replica_config.gradient_clipping:
                    gradients = clip_gradients(gradients, replica_config.max_gradient_norm)
                
                if accrued_gradients is None:
                    accrued_gradients = gradients.clone()
                else:
                    accrued_gradients += gradients
                
                # 4. Push gradientes cada n_push steps
                if step % replica_config.push_frequency == 0:
                    if accrued_gradients is not None:
                        _async_push_gradients(
                            replica_id, transport, total_params,
                            num_shards, accrued_gradients
                        )
                        accrued_gradients = None
                
                # 5. Actualizar step global
                with global_step.get_lock():
                    global_step.value += 1
                    current_global_step = global_step.value
                
                # 6. Señalizar warm start (una sola vez)
                if is_warm_start and not warm_start_already_signaled \
                        and current_global_step >= replica_config.warm_start_steps:
                    warm_start_done.value = True
                    warm_start_already_signaled = True
                    logger.info(
                        f"Warm start completado en step {current_global_step}. "
                        f"Activando réplicas restantes."
                    )
                
                step += 1
                
                # Log con loss y accuracy
                if step % replica_config.log_frequency == 0:
                    logger.info(
                        f"Replica {replica_id} | "
                        f"Step {step} (global: {current_global_step}) | "
                        f"Loss: {loss.item():.4f} | "
                        f"Acc: {batch_accuracy:.2f}%"
                    )
            
            epoch += 1
            logger.info(f"Replica {replica_id}: Epoch {epoch} completada")
    
    except Exception as e:
        logger.error(f"Error en replica {replica_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if accrued_gradients is not None:
            _async_push_gradients(
                replica_id, transport, total_params,
                num_shards, accrued_gradients
            )
        
        transport.cleanup()
        
        logger.info(f"Replica {replica_id} detenida. Steps completados: {step}")


def _consume_parameters_from_ps(
    transport: ReplicaTransport,
    model: nn.Module,
    total_params: int,
    device: torch.device
):
    """Consume TODOS los mensajes del PS. Actualiza params, descarta ACKs."""
    params = torch.zeros(total_params)
    updated = False
    
    while True:
        try:
            response = transport.receive(block=False, timeout=0.01)
            if response is None:
                break
            
            if response.msg_type == MessageType.PARAMETERS_RESPONSE:
                start_idx = response.data['start_idx']
                end_idx = response.data['end_idx']
                shard_params = response.data['parameters']
                params[start_idx:end_idx] = shard_params
                updated = True
            # else: ACK u otro -> DESCARTAR (no re-encolar)
            
        except queue.Empty:
            break
        except Exception:
            break
    
    if updated:
        model.set_flat_parameters(params.to(device))


def _setup_replica_logging(replica_id: int):
    """Configura logging: consola + archivo para auditoría."""
    log_dir = os.environ.get("DISTBELIEF_LOG_DIR", "./logs")
    log_file = os.environ.get("DISTBELIEF_LOG_FILE", "distbelief_training.log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    _logger = logging.getLogger("ModelReplica")
    
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)


def _fetch_all_parameters(
    replica_id: int,
    transport: ReplicaTransport,
    model: nn.Module,
    total_params: int,
    num_shards: int,
    device: torch.device
) -> bool:
    """Fetch inicial de todos los parámetros del PS (síncrono)."""
    for shard_id in range(num_shards):
        msg = Message(
            msg_type=MessageType.GET_PARAMETERS,
            sender_id=replica_id,
            data={}
        )
        transport.send_to_shard(shard_id, msg)
    
    params = torch.zeros(total_params)
    received_shards = set()
    
    timeout = time.time() + 10.0
    while len(received_shards) < num_shards and time.time() < timeout:
        response = transport.receive(block=True, timeout=0.5)
        if response and response.msg_type == MessageType.PARAMETERS_RESPONSE:
            shard_id = response.data['shard_id']
            start_idx = response.data['start_idx']
            end_idx = response.data['end_idx']
            shard_params = response.data['parameters']
            params[start_idx:end_idx] = shard_params
            received_shards.add(shard_id)
    
    if len(received_shards) > 0:
        model.set_flat_parameters(params.to(device))
        logger.info(
            f"Replica {replica_id}: Parámetros cargados "
            f"({len(received_shards)}/{num_shards} shards)"
        )
        return True
    return False


def _async_fetch_parameters(
    replica_id: int,
    transport: ReplicaTransport,
    num_shards: int
):
    """Envía requests asíncronos para fetchear parámetros."""
    for shard_id in range(num_shards):
        msg = Message(
            msg_type=MessageType.GET_PARAMETERS,
            sender_id=replica_id,
            data={}
        )
        transport.send_to_shard(shard_id, msg)


def _async_push_gradients(
    replica_id: int,
    transport: ReplicaTransport,
    total_params: int,
    num_shards: int,
    gradients: torch.Tensor
):
    """Envía gradientes acumulados al PS de forma asíncrona."""
    shard_grads = split_gradients_by_shard(gradients, total_params, num_shards)
    
    for shard_id, (start, end, grad_shard) in shard_grads.items():
        msg = Message(
            msg_type=MessageType.PUSH_GRADIENTS,
            sender_id=replica_id,
            data={
                'shard_id': shard_id,
                'start_idx': start,
                'end_idx': end,
                'gradients': grad_shard
            }
        )
        transport.send_to_shard(shard_id, msg)


def create_data_splits(num_examples: int, num_replicas: int,
                       seed: int = 42) -> List[List[int]]:
    """Divide los datos entre las réplicas."""
    import numpy as np
    np.random.seed(seed)
    indices = np.random.permutation(num_examples)
    splits = np.array_split(indices, num_replicas)
    return [split.tolist() for split in splits]
