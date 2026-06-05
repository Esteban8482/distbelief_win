"""
Utilidades para comunicación entre procesos y operaciones auxiliares.

Implementa el mecanismo de comunicación entre Parameter Server shards 
y Model Replicas usando multiprocessing de Python de forma compatible
con Windows (spawn mode).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp
from multiprocessing import Queue, Process, Lock
import queue
import time
import pickle
import logging
from dataclasses import dataclass
from enum import Enum


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MessageType(Enum):
    """Tipos de mensajes para comunicación entre procesos."""
    # Replica -> PS
    GET_PARAMETERS = "get_parameters"
    PUSH_GRADIENTS = "push_gradients"
    
    # PS -> Replica
    PARAMETERS_RESPONSE = "parameters_response"
    GRADIENTS_ACK = "gradients_ack"
    
    # Control
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    GET_STATS = "get_stats"
    STATS_RESPONSE = "stats_response"
    
    # Checkpoint
    SAVE_CHECKPOINT = "save_checkpoint"
    LOAD_CHECKPOINT = "load_checkpoint"
    CHECKPOINT_DONE = "checkpoint_done"


@dataclass
class Message:
    """Mensaje para comunicación entre procesos."""
    msg_type: MessageType
    sender_id: int
    data: Any = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


def get_parameter_shard_bounds(total_params: int, num_shards: int, 
                                shard_id: int) -> Tuple[int, int]:
    """
    Calcula los límites de un shard de parámetros.
    
    Según el paper (Sección 4.1): "if we have 10 parameter server shards, 
    each shard is responsible for storing and applying updates to 1/10th 
    of the model parameters"
    
    Args:
        total_params: Número total de parámetros
        num_shards: Número total de shards
        shard_id: ID del shard (0-indexed)
        
    Returns:
        Tuple (start_idx, end_idx) con los límites del shard
    """
    shard_size = total_params // num_shards
    remainder = total_params % num_shards
    
    # Distribuir el remainder entre los primeros shards
    if shard_id < remainder:
        start = shard_id * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = remainder * (shard_size + 1) + (shard_id - remainder) * shard_size
        end = start + shard_size
    
    return start, end


def get_shard_for_parameter(param_idx: int, total_params: int, 
                            num_shards: int) -> int:
    """
    Determina qué shard maneja un parámetro dado su índice global.
    
    Args:
        param_idx: Índice global del parámetro
        total_params: Número total de parámetros
        num_shards: Número de shards
        
    Returns:
        ID del shard responsable
    """
    for shard_id in range(num_shards):
        start, end = get_parameter_shard_bounds(total_params, num_shards, shard_id)
        if start <= param_idx < end:
            return shard_id
    return num_shards - 1


def split_gradients_by_shard(flat_gradients: torch.Tensor, total_params: int,
                             num_shards: int) -> Dict[int, Tuple[int, int, torch.Tensor]]:
    """
    Divide los gradientes en chunks correspondientes a cada shard.
    
    Args:
        flat_gradients: Gradientes aplanados
        total_params: Número total de parámetros
        num_shards: Número de shards
        
    Returns:
        Dict: shard_id -> (start_idx, end_idx, gradient_shard)
    """
    shards = {}
    for shard_id in range(num_shards):
        start, end = get_parameter_shard_bounds(total_params, num_shards, shard_id)
        shards[shard_id] = (start, end, flat_gradients[start:end].clone())
    return shards


def clip_gradients(gradients: torch.Tensor, max_norm: float) -> torch.Tensor:
    """
    Aplica gradient clipping.
    
    Args:
        gradients: Tensor de gradientes
        max_norm: Norma máxima permitida
        
    Returns:
        Gradientes clippeados
    """
    norm = gradients.norm()
    if norm > max_norm:
        gradients = gradients * (max_norm / norm)
    return gradients


class Timer:
    """Timer para medir tiempos de ejecución."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    def get_elapsed(self) -> float:
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed


def count_parameters(model: nn.Module) -> int:
    """Cuenta el número total de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_layers_info(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Obtiene información sobre las capas del modelo.
    
    Returns:
        Lista de diccionarios con nombre, shape y número de parámetros
    """
    layers = []
    for name, param in model.named_parameters():
        layers.append({
            'name': name,
            'shape': list(param.shape),
            'num_params': param.numel(),
            'trainable': param.requires_grad
        })
    return layers


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """
    Aplana todos los parámetros del modelo en un solo tensor 1D.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Tensor 1D con todos los parámetros concatenados
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def unflatten_parameters(model: nn.Module, flat_params: torch.Tensor):
    """
    Restaura los parámetros del modelo desde un tensor 1D.
    
    Args:
        model: Modelo PyTorch
        flat_params: Tensor 1D con parámetros aplanados
    """
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(
            flat_params[offset:offset + numel].view_as(param)
        )
        offset += numel
