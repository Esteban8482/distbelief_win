"""
Parameter Server con Sharding y Adagrad.

Implementa el Parameter Server central descrito en el paper:
"We divide the training data into a number of subsets and run a copy 
of the model on each of these subsets. The models communicate updates 
through a centralized parameter server, which keeps the current state 
of all parameters for the model, sharded across many machines"

Características implementadas:
- Sharding de parámetros entre múltiples shards
- Adagrad adaptativo por parámetro
- Operaciones asíncronas de GET y PUSH
- Checkpointing
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time
import os
import pickle
import logging

from utils import (
    Message, MessageType,
    get_parameter_shard_bounds
)
from config import ParameterServerConfig


logger = logging.getLogger("ParameterServer")


class ParameterShard:
    """
    Un shard del Parameter Server.
    
    Según el paper (Sección 4.1): "the parameter server shards also 
    run independently of one another"
    
    Cada shard:
    - Almacena una fracción de los parámetros del modelo
    - Mantiene el estado de Adagrad (suma de gradientes cuadrados)
    - Aplica updates de forma independiente
    """
    
    def __init__(self, shard_id: int, start_idx: int, end_idx: int,
                 config: ParameterServerConfig):
        """
        Args:
            shard_id: ID único del shard
            start_idx: Índice inicial de los parámetros (global)
            end_idx: Índice final de los parámetros (global)
            config: Configuración del PS
        """
        self.shard_id = shard_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.config = config
        self.size = end_idx - start_idx
        
        # Parámetros almacenados en este shard (inicializados luego)
        self.parameters = None
        
        # Estado de Adagrad: suma acumulada de gradientes cuadrados
        # η_{i,K} = γ / sqrt(Σ_{j=1}^{K} Δw_{i,j}^2)
        # Según el paper: "Adagrad uses a separate adaptive learning rate 
        # for each parameter... these learning rates are computed only from 
        # the summed squared gradients of each parameter"
        self.accumulated_squared_gradients = None
        
        # Contadores de updates
        self.num_updates = 0
        self.num_get_requests = 0
        
        self.logger = logging.getLogger(f"Shard_{shard_id}")
        self.logger.info(
            f"Shard {shard_id} inicializado: parámetros "
            f"[{start_idx}:{end_idx}] (total: {self.size})"
        )
    
    def initialize_parameters(self, initial_values: torch.Tensor):
        """
        Inicializa los parámetros con valores dados.
        
        Args:
            initial_values: Tensor con los valores iniciales para este shard
        """
        assert len(initial_values) == self.size, \
            f"Tamaño mismatch: {len(initial_values)} vs {self.size}"
        
        self.parameters = initial_values.clone()
        
        # Inicializar acumuladores de Adagrad en cero
        self.accumulated_squared_gradients = torch.zeros(self.size)
        
        self.logger.info(
            f"Parámetros inicializados. Shape: {self.parameters.shape}"
        )
    
    def get_parameters(self) -> torch.Tensor:
        """
        Retorna los parámetros actuales de este shard.
        
        Returns:
            Tensor con los parámetros [start_idx:end_idx]
        """
        self.num_get_requests += 1
        return self.parameters.clone()
    
    def apply_gradients(self, gradients: torch.Tensor):
        """
        Aplica gradientes usando Adagrad o SGD con learning rate fijo.
        
        Según el paper (Sección 4.1):
        - Adagrad: η_{i,K} = γ / sqrt(Σ_{j=1}^{K} Δw_{i,j}^2)
        - "Adagrad is easily implemented locally within each parameter server shard"
        
        Args:
            gradients: Gradientes para los parámetros de este shard
        """
        assert len(gradients) == self.size, \
            f"Tamaño de gradientes mismatch: {len(gradients)} vs {self.size}"
        
        if self.config.use_adagrad:
            # Adagrad adaptativo
            # Actualizar acumulador de gradientes cuadrados
            self.accumulated_squared_gradients += gradients ** 2
            
            # Calcular learning rate adaptativo por parámetro
            # η_i = γ / sqrt(accumulated_g² + ε)
            adaptive_lr = self.config.adagrad_gamma / (
                torch.sqrt(self.accumulated_squared_gradients) + 
                self.config.adagrad_epsilon
            )
            
            # Aplicar update: w = w - η * gradient
            self.parameters -= adaptive_lr * gradients
        else:
            # SGD con learning rate fijo
            self.parameters -= self.config.base_learning_rate * gradients
        
        self.num_updates += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del shard."""
        return {
            'shard_id': self.shard_id,
            'size': self.size,
            'num_updates': self.num_updates,
            'num_get_requests': self.num_get_requests,
            'param_mean': self.parameters.mean().item() if self.parameters is not None else 0,
            'param_std': self.parameters.std().item() if self.parameters is not None else 0,
        }


def parameter_server_shard_process(
    shard_id: int,
    request_queue: Queue,
    response_queue: Queue,
    should_stop,
    config_dict: dict,
    total_params: int,
    initial_parameters: Optional[torch.Tensor] = None
):
    """
    Proceso para un shard del Parameter Server.
    
    Este proceso corre de forma independiente, escuchando requests
    de las model replicas y respondiendo de forma asíncrona.
    
    Args:
        shard_id: ID del shard
        request_queue: Cola para recibir requests
        response_queue: Cola para enviar responses
        should_stop: Value compartido para señal de parada
        config_dict: Configuración serializada como dict
        total_params: Número total de parámetros del modelo
        initial_parameters: Parámetros iniciales (si None, se inicializan en cero)
    """
    # Recrear configuración desde dict
    config = ParameterServerConfig(**config_dict)
    
    logger.info(f"Iniciando PS shard {shard_id}")
    
    # Calcular límites del shard
    start_idx, end_idx = get_parameter_shard_bounds(
        total_params, config.num_shards, shard_id
    )
    
    # Crear el shard
    shard = ParameterShard(shard_id, start_idx, end_idx, config)
    
    # Inicializar parámetros
    if initial_parameters is not None:
        shard_params = initial_parameters[start_idx:end_idx].clone()
    else:
        shard_params = torch.zeros(end_idx - start_idx)
    
    shard.initialize_parameters(shard_params)
    
    logger.info(f"PS shard {shard_id} listo para procesar requests")
    
    # Loop principal: procesar requests de forma asíncrona
    while not should_stop.value:
        try:
            # Recibir request de una replica (blocking con timeout)
            try:
                msg = request_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue
            
            if msg.msg_type == MessageType.GET_PARAMETERS:
                # Responder con los parámetros actuales
                params = shard.get_parameters()
                response = Message(
                    msg_type=MessageType.PARAMETERS_RESPONSE,
                    sender_id=shard_id,
                    data={
                        'shard_id': shard_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'parameters': params
                    }
                )
                response_queue.put(response)
                
            elif msg.msg_type == MessageType.PUSH_GRADIENTS:
                # Aplicar gradientes
                grad_data = msg.data
                gradients = grad_data['gradients']
                
                # Aplicar al shard
                shard.apply_gradients(gradients)
                
                # Acknowledge
                ack = Message(
                    msg_type=MessageType.GRADIENTS_ACK,
                    sender_id=shard_id,
                    data={'shard_id': shard_id, 'status': 'applied'}
                )
                response_queue.put(ack)
                
            elif msg.msg_type == MessageType.GET_STATS:
                # Enviar estadísticas
                stats = shard.get_stats()
                response = Message(
                    msg_type=MessageType.STATS_RESPONSE,
                    sender_id=shard_id,
                    data=stats
                )
                response_queue.put(response)
                
            elif msg.msg_type == MessageType.SAVE_CHECKPOINT:
                # Guardar checkpoint
                checkpoint_data = {
                    'shard_id': shard_id,
                    'parameters': shard.parameters.clone(),
                    'accumulated_squared_gradients': 
                        shard.accumulated_squared_gradients.clone(),
                    'num_updates': shard.num_updates
                }
                filepath = os.path.join(
                    config.checkpoint_dir, 
                    f"shard_{shard_id}_checkpoint.pkl"
                )
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                with open(filepath, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                ack = Message(
                    msg_type=MessageType.CHECKPOINT_DONE,
                    sender_id=shard_id,
                    data={'filepath': filepath}
                )
                response_queue.put(ack)
                
            elif msg.msg_type == MessageType.LOAD_CHECKPOINT:
                # Cargar checkpoint
                filepath = os.path.join(
                    config.checkpoint_dir,
                    f"shard_{shard_id}_checkpoint.pkl"
                )
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    shard.parameters = checkpoint_data['parameters']
                    shard.accumulated_squared_gradients = \
                        checkpoint_data['accumulated_squared_gradients']
                    shard.num_updates = checkpoint_data['num_updates']
                
                ack = Message(
                    msg_type=MessageType.CHECKPOINT_DONE,
                    sender_id=shard_id,
                    data={'loaded': os.path.exists(filepath)}
                )
                response_queue.put(ack)
                
        except Exception as e:
            logger.error(f"Error en PS shard {shard_id}: {e}")
    
    logger.info(f"PS shard {shard_id} detenido. "
                f"Updates procesados: {shard.num_updates}")


class ParameterServerCoordinator:
    """
    Coordinador del Parameter Server.
    
    Gestiona múltiples shards del PS y provee una interfaz unificada
    para las model replicas.
    """
    
    def __init__(self, config: ParameterServerConfig, total_params: int):
        """
        Args:
            config: Configuración del PS
            total_params: Número total de parámetros
        """
        self.config = config
        self.total_params = total_params
        
        # Crear colas de comunicación en el proceso principal
        self.request_queues = [Queue() for _ in range(config.num_shards)]
        self.response_queue = Queue()
        self.should_stop = mp.Value('b', False)
        self.shard_processes = []
        
        # Serializar configuración para pasar a los procesos
        self.config_dict = {
            'num_shards': config.num_shards,
            'base_port': config.base_port,
            'checkpoint_dir': config.checkpoint_dir,
            'checkpoint_frequency': config.checkpoint_frequency,
            'adagrad_gamma': config.adagrad_gamma,
            'adagrad_epsilon': config.adagrad_epsilon,
            'use_adagrad': config.use_adagrad,
            'base_learning_rate': config.base_learning_rate,
        }
        
        logger.info(
            f"PS Coordinator inicializado: "
            f"{config.num_shards} shards, "
            f"{total_params:,} parámetros totales"
        )
    
    def start(self, initial_model_state: Optional[Dict] = None):
        """
        Inicia los procesos de los shards.
        
        Args:
            initial_model_state: Estado inicial del modelo (opcional)
        """
        # Extraer parámetros iniciales si se proporcionan
        initial_params = None
        if initial_model_state is not None:
            initial_params = initial_model_state.get('parameters')
        
        # Crear y lanzar procesos para cada shard
        for shard_id in range(self.config.num_shards):
            start_idx, end_idx = get_parameter_shard_bounds(
                self.total_params, self.config.num_shards, shard_id
            )
            
            # Preparar parámetros iniciales para este shard
            shard_initial = None
            if initial_params is not None:
                shard_initial = initial_params[start_idx:end_idx]
            
            p = mp.Process(
                target=parameter_server_shard_process,
                args=(
                    shard_id, 
                    self.request_queues[shard_id],
                    self.response_queue,
                    self.should_stop,
                    self.config_dict,
                    self.total_params, 
                    shard_initial
                )
            )
            p.start()
            self.shard_processes.append(p)
            
            logger.info(
                f"Shard {shard_id} lanzado: "
                f"parámetros [{start_idx}:{end_idx}]"
            )
        
        logger.info(
            f"Todos los {self.config.num_shards} shards iniciados"
        )
    
    def stop(self):
        """Detiene todos los shards del PS."""
        logger.info("Deteniendo Parameter Server...")
        self.should_stop.value = True
        
        for i, p in enumerate(self.shard_processes):
            p.join(timeout=5.0)
            if p.is_alive():
                logger.warning(f"Shard {i} no respondió, forzando terminación")
                p.terminate()
        
        logger.info("Parameter Server detenido")
    
    def get_request_queue(self, shard_id: int) -> Queue:
        """Retorna la cola de requests para un shard."""
        return self.request_queues[shard_id]
    
    def get_response_queue(self) -> Queue:
        """Retorna la cola de responses."""
        return self.response_queue
    
    def save_checkpoint(self):
        """Solicita a todos los shards que guarden su estado."""
        logger.info("Guardando checkpoint...")
        for shard_id in range(self.config.num_shards):
            msg = Message(
                msg_type=MessageType.SAVE_CHECKPOINT,
                sender_id=-1,  # Coordinator
                data={}
            )
            self.request_queues[shard_id].put(msg)
        
        # Esperar confirmaciones
        acks = 0
        while acks < self.config.num_shards:
            try:
                response = self.response_queue.get(block=True, timeout=5.0)
                if response.msg_type == MessageType.CHECKPOINT_DONE:
                    acks += 1
            except queue.Empty:
                break
        
        logger.info(f"Checkpoint guardado ({acks}/{self.config.num_shards} shards)")
    
    def load_checkpoint(self):
        """Solicita a todos los shards que carguen su estado."""
        logger.info("Cargando checkpoint...")
        for shard_id in range(self.config.num_shards):
            msg = Message(
                msg_type=MessageType.LOAD_CHECKPOINT,
                sender_id=-1,
                data={}
            )
            self.request_queues[shard_id].put(msg)
        
        # Esperar confirmaciones
        acks = 0
        while acks < self.config.num_shards:
            try:
                response = self.response_queue.get(block=True, timeout=5.0)
                if response.msg_type == MessageType.CHECKPOINT_DONE:
                    acks += 1
            except queue.Empty:
                break
        
        logger.info(f"Checkpoint cargado ({acks}/{self.config.num_shards} shards)")
    
    def get_stats(self) -> List[Dict]:
        """Obtiene estadísticas de todos los shards."""
        # Solicitar stats a cada shard
        for shard_id in range(self.config.num_shards):
            msg = Message(
                msg_type=MessageType.GET_STATS,
                sender_id=-1,
                data={}
            )
            self.request_queues[shard_id].put(msg)
        
        # Recolectar respuestas
        stats = []
        for _ in range(self.config.num_shards):
            try:
                response = self.response_queue.get(block=True, timeout=2.0)
                if response.msg_type == MessageType.STATS_RESPONSE:
                    stats.append(response.data)
            except queue.Empty:
                break
        
        return stats
