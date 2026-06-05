"""
Model Replica con Downpour SGD asíncrono.

Implementa el algoritmo Downpour SGD descrito en el paper:
"before processing each mini-batch, a model replica asks the parameter 
server service for an updated copy of its model parameters... processes 
a mini-batch of data to compute a parameter gradient, and sends the 
gradient to the parameter server"

CORRECCIONES v2:
- Fix memory leak: ACKs y mensajes no-PARAMETERS_RESPONSE se descartan
  en vez de re-encolarse infinitamente
- Fix convergencia: la réplica NO aplica SGD local. Solo fetchea params,
  computa gradientes, y envía al PS. El PS (con Adagrad) es la única
  fuente de verdad para los parámetros.
- Logging con accuracy a consola y archivo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Any
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


logger = logging.getLogger("ModelReplica")


def model_replica_process(
    replica_id: int,
    ps_request_queues: List[Queue],
    ps_response_queue: Queue,
    should_stop,
    warm_start_done,
    global_step,
    replica_config_dict: dict,
    ps_config_dict: dict,
    total_params: int,
    train_data_indices: List[int],
    data_dir: str = "./data",
    is_warm_start: bool = False
):
    """
    Proceso para una Model Replica con Downpour SGD.
    
    DISEÑO CORREGIDO (v2):
    
    Downpour SGD según Dean et al. (2012):
    
    1. Fetchear parámetros w del Parameter Server (cada n_fetch steps)
    2. Procesar mini-batch: computar loss y gradientes ∇L(w, data)
    3. Enviar gradientes al PS (cada n_push steps)
    4. NO actualizar parámetros localmente — el PS es la única
       fuente de verdad. Los parámetros locales se sobreescriben
       en el próximo fetch.
    
    El PS aplica: w ← w − η_adagrad · ∇L
    
    La réplica aplica: nada. Solo computa y envía gradientes.
    
    Esto evita:
    - Conflicto entre SGD local (lr=0.01) y Adagrad global
    - Divergencia entre réplicas
    - Staleness de parámetros localmente modificados
    """
    # Recrear configuraciones
    replica_config = ModelReplicaConfig(**replica_config_dict)
    ps_config = ParameterServerConfig(**ps_config_dict)
    
    # Configurar logging a archivo para auditoría del entrenamiento
    _setup_replica_logging(replica_id)
    
    logger.info(f"Iniciando Model Replica {replica_id} "
                f"(warm_start={is_warm_start})")
    
    # Configurar device
    device = torch.device(replica_config.device)
    
    # Crear modelo
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Inicializar parámetros desde el PS (fetch inicial)
    logger.info(f"Replica {replica_id}: Fetch inicial de parámetros")
    _fetch_all_parameters(
        replica_id, ps_request_queues, ps_response_queue,
        model, total_params, ps_config.num_shards, device
    )
    
    # Preparar dataset (CIFAR-10)
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
    
    # Subset para esta replica
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
    
    # Loop de entrenamiento - Downpour SGD puro
    logger.info(f"Replica {replica_id}: Comenzando entrenamiento")
    
    step = 0
    accrued_gradients = None
    
    # Flag local para evitar spam del mensaje de warm start
    warm_start_already_signaled = False
    
    try:
        epoch = 0
        max_epochs = 50
        
        while not should_stop.value and epoch < max_epochs:
            for batch_idx, (data, target) in enumerate(dataloader):
                if should_stop.value:
                    break
                
                # Verificar si estamos en warm start
                if not is_warm_start and not warm_start_done.value:
                    time.sleep(0.5)
                    continue
                
                data, target = data.to(device), target.to(device)
                
                # ===== Downpour SGD Step (corregido) =====
                
                # 1. Fetch parámetros cada n_fetch steps (asíncrono)
                if step % replica_config.fetch_frequency == 0:
                    _async_fetch_parameters(
                        replica_id, ps_request_queues, ps_config.num_shards
                    )
                
                # Consumir parámetros del PS (sin re-encolar ACKs)
                _consume_parameters_from_ps(
                    ps_response_queue, model, total_params, device
                    # FIX v2: consume TODOS los mensajes, descarta no-PARAMS
                )
                
                # 2. Computar gradiente (forward + backward)
                # NOTA: NO usamos optimizer local. El PS actualiza los params.
                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Calcular accuracy del batch para logging
                with torch.no_grad():
                    _, predicted = output.max(1)
                    correct = predicted.eq(target).sum().item()
                    total = target.size(0)
                    batch_accuracy = 100.0 * correct / total
                
                # 3. Obtener gradientes y acumular
                gradients = model.get_flat_gradients()
                
                # Clip gradients
                if replica_config.gradient_clipping:
                    gradients = clip_gradients(gradients, replica_config.max_gradient_norm)
                
                if accrued_gradients is None:
                    accrued_gradients = gradients.clone()
                else:
                    accrued_gradients += gradients
                
                # 4. Push gradientes cada n_push steps (asíncrono)
                if step % replica_config.push_frequency == 0:
                    if accrued_gradients is not None:
                        _async_push_gradients(
                            replica_id, ps_request_queues, total_params,
                            ps_config.num_shards, accrued_gradients
                        )
                        accrued_gradients = None
                
                # 5. Actualizar step global
                with global_step.get_lock():
                    global_step.value += 1
                    current_global_step = global_step.value
                
                # 6. Señalizar fin de warm start (una sola vez)
                if is_warm_start and not warm_start_already_signaled \
                        and current_global_step >= replica_config.warm_start_steps:
                    warm_start_done.value = True
                    warm_start_already_signaled = True
                    logger.info(
                        f"Warm start completado en step {current_global_step}. "
                        f"Activando réplicas restantes."
                    )
                
                step += 1
                
                # Log periódico con loss y accuracy
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
        # Push gradientes restantes
        if accrued_gradients is not None:
            _async_push_gradients(
                replica_id, ps_request_queues, total_params,
                ps_config.num_shards, accrued_gradients
            )
        
        logger.info(
            f"Replica {replica_id} detenida. "
            f"Steps completados: {step}"
        )


def _consume_parameters_from_ps(
    ps_response_queue: Queue,
    model: nn.Module,
    total_params: int,
    device: torch.device
):
    """
    FIX v2: Consume TODOS los mensajes pendientes de la response queue.
    
    Problema anterior (v1):
        Los mensajes no-PARAMETERS_RESPONSE (ACKs, etc.) se re-encolaban,
        causando un loop infinito de lectura/re-encolación y memory leak.
    
    Solución (v2):
        Consumir TODOS los mensajes disponibles sin bloquear.
        - PARAMETERS_RESPONSE: actualizar parámetros del modelo
        - Cualquier otro tipo: descartar (no re-encolar)
    
    Esto garantiza que la queue nunca crece indefinidamente.
    """
    params = torch.zeros(total_params)
    updated = False
    
    # Consumir TODOS los mensajes disponibles (non-blocking)
    while True:
        try:
            response = ps_response_queue.get(block=False)
            
            if response.msg_type == MessageType.PARAMETERS_RESPONSE:
                # Actualizar los parámetros correspondientes al shard
                shard_id = response.data['shard_id']
                start_idx = response.data['start_idx']
                end_idx = response.data['end_idx']
                shard_params = response.data['parameters']
                params[start_idx:end_idx] = shard_params
                updated = True
            # else: descartar el mensaje (ACKs, etc.)
            
        except queue.Empty:
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
    ps_request_queues: List[Queue],
    ps_response_queue: Queue,
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
        ps_request_queues[shard_id].put(msg)
    
    params = torch.zeros(total_params)
    received_shards = set()
    
    timeout = time.time() + 10.0
    while len(received_shards) < num_shards and time.time() < timeout:
        try:
            response = ps_response_queue.get(block=True, timeout=0.5)
            if (response and 
                response.msg_type == MessageType.PARAMETERS_RESPONSE):
                shard_id = response.data['shard_id']
                start_idx = response.data['start_idx']
                end_idx = response.data['end_idx']
                shard_params = response.data['parameters']
                params[start_idx:end_idx] = shard_params
                received_shards.add(shard_id)
        except queue.Empty:
            continue
    
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
    ps_request_queues: List[Queue],
    num_shards: int
):
    """Envía requests asíncronos para fetchear parámetros a todos los shards."""
    for shard_id in range(num_shards):
        msg = Message(
            msg_type=MessageType.GET_PARAMETERS,
            sender_id=replica_id,
            data={}
        )
        try:
            ps_request_queues[shard_id].put(msg, block=False)
        except queue.Full:
            pass


def _async_push_gradients(
    replica_id: int,
    ps_request_queues: List[Queue],
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
        try:
            ps_request_queues[shard_id].put(msg, block=False)
        except queue.Full:
            pass


def create_data_splits(num_examples: int, num_replicas: int, 
                       seed: int = 42) -> List[List[int]]:
    """Divide los datos entre las réplicas."""
    import numpy as np
    
    np.random.seed(seed)
    indices = np.random.permutation(num_examples)
    splits = np.array_split(indices, num_replicas)
    
    return [split.tolist() for split in splits]
