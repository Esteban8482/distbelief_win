"""
Parameter Server con Sharding y Adagrad.

Soporta dos modos de operación:
- LOCAL: multiprocessing.Queue (todo en una máquina)
- DISTRIBUIDO: TCP/IP (PS en una máquina, réplicas en otras)

Implementa el Parameter Server central descrito en el paper:
"The models communicate updates through a centralized parameter server,
which keeps the current state of all parameters for the model, sharded
across many machines"
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
import threading

from utils import (
    Message, MessageType,
    get_parameter_shard_bounds
)
from config import ParameterServerConfig
from networking import (
    ParameterServerListener, _send_message, _recv_message,
    get_local_ip
)


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
        self.shard_id = shard_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.config = config
        self.size = end_idx - start_idx
        
        self.parameters = None
        
        # Estado de Adagrad: η_{i,K} = γ / sqrt(Σ Δw_{i,j}^2)
        self.accumulated_squared_gradients = None
        
        self.num_updates = 0
        self.num_get_requests = 0
        
        self.logger = logging.getLogger(f"Shard_{shard_id}")
        self.logger.info(
            f"Shard {shard_id} inicializado: parámetros "
            f"[{start_idx}:{end_idx}] (total: {self.size})"
        )
    
    def initialize_parameters(self, initial_values: torch.Tensor):
        """Inicializa los parámetros con valores dados."""
        assert len(initial_values) == self.size, \
            f"Tamaño mismatch: {len(initial_values)} vs {self.size}"
        
        self.parameters = initial_values.clone()
        self.accumulated_squared_gradients = torch.zeros(self.size)
        
        self.logger.info(
            f"Parámetros inicializados. Shape: {self.parameters.shape}"
        )
    
    def get_parameters(self) -> torch.Tensor:
        """Retorna los parámetros actuales de este shard."""
        self.num_get_requests += 1
        return self.parameters.clone()
    
    def apply_gradients(self, gradients: torch.Tensor):
        """
        Aplica gradientes usando Adagrad o SGD con learning rate fijo.
        
        Según el paper (Sección 4.1):
        - Adagrad: η_{i,K} = γ / sqrt(Σ_{j=1}^{K} Δw_{i,j}^2)
        - "Adagrad is easily implemented locally within each parameter server shard"
        """
        assert len(gradients) == self.size, \
            f"Tamaño de gradientes mismatch: {len(gradients)} vs {self.size}"
        
        if self.config.use_adagrad:
            # Adagrad adaptativo
            self.accumulated_squared_gradients += gradients ** 2
            
            adaptive_lr = self.config.adagrad_gamma / (
                torch.sqrt(self.accumulated_squared_gradients) + 
                self.config.adagrad_epsilon
            )
            
            self.parameters -= adaptive_lr * gradients
        else:
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
    request_queue: Optional[Queue],
    response_queues: Optional[Dict[int, Queue]],
    listener_host: Optional[str],
    listener_port: Optional[int],
    should_stop,
    config_dict: dict,
    total_params: int,
    initial_parameters: Optional[torch.Tensor] = None
):
    """
    Proceso para un shard del Parameter Server.
    
    Funciona en dos modos:
    - LOCAL: request_queue y response_queues son multiprocessing.Queue
    - DISTRIBUIDO: listener_host/port especifican dónde escuchar TCP
    
    Args:
        shard_id: ID del shard
        request_queue: Cola para recibir requests (modo local) o None
        response_queues: Dict de colas para enviar responses (modo local) o None
        listener_host: Host para escuchar TCP (modo distribuido) o None
        listener_port: Puerto TCP (modo distribuido) o None
        should_stop: Value compartido para señal de parada
        config_dict: Configuración serializada como dict
        total_params: Número total de parámetros del modelo
        initial_parameters: Parámetros iniciales
    """
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
    
    # --- MODO DISTRIBUIDO: escuchar conexiones TCP ---
    tcp_listener = None
    if listener_host is not None and listener_port is not None:
        # Cola local para requests de réplicas conectadas vía TCP
        local_request_queue = queue.Queue()
        local_response_queues: Dict[int, queue.Queue] = {}
        
        tcp_listener = ParameterServerListener(
            host=listener_host,
            port=listener_port,
            request_queue=local_request_queue,
            response_queues=local_response_queues
        )
        tcp_listener.start()
        
        # Usar las colas locales como fuente/sink
        request_queue = local_request_queue
        response_queues = local_response_queues
        
        logger.info(f"PS shard {shard_id} escuchando en TCP {listener_host}:{listener_port}")
    
    logger.info(f"PS shard {shard_id} listo para procesar requests")
    
    # Loop principal: procesar requests de forma asíncrona
    while not should_stop.value:
        try:
            # Recibir request (blocking con timeout)
            try:
                msg = request_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue
            
            sender_id = getattr(msg, 'sender_id', -1)
            
            if msg.msg_type == MessageType.GET_PARAMETERS:
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
                
                # Enviar response al sender correcto
                if sender_id in response_queues:
                    try:
                        response_queues[sender_id].put(response, block=False)
                    except queue.Full:
                        pass
                elif response_queues:
                    # Fallback: enviar a la primera cola disponible
                    try:
                        next(iter(response_queues.values())).put(response, block=False)
                    except queue.Full:
                        pass
                
            elif msg.msg_type == MessageType.PUSH_GRADIENTS:
                grad_data = msg.data
                gradients = grad_data['gradients']
                shard.apply_gradients(gradients)
                
                ack = Message(
                    msg_type=MessageType.GRADIENTS_ACK,
                    sender_id=shard_id,
                    data={'shard_id': shard_id, 'status': 'applied'}
                )
                
                if sender_id in response_queues:
                    try:
                        response_queues[sender_id].put(ack, block=False)
                    except queue.Full:
                        pass
                
            elif msg.msg_type == MessageType.GET_STATS:
                stats = shard.get_stats()
                response = Message(
                    msg_type=MessageType.STATS_RESPONSE,
                    sender_id=shard_id,
                    data=stats
                )
                
                if sender_id in response_queues:
                    try:
                        response_queues[sender_id].put(response, block=False)
                    except queue.Full:
                        pass
                
            elif msg.msg_type == MessageType.SAVE_CHECKPOINT:
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
                
                if sender_id in response_queues:
                    try:
                        response_queues[sender_id].put(ack, block=False)
                    except queue.Full:
                        pass
                
            elif msg.msg_type == MessageType.LOAD_CHECKPOINT:
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
                
                if sender_id in response_queues:
                    try:
                        response_queues[sender_id].put(ack, block=False)
                    except queue.Full:
                        pass
                
        except Exception as e:
            logger.error(f"Error en PS shard {shard_id}: {e}")
    
    # Cleanup
    if tcp_listener:
        tcp_listener.stop()
    
    logger.info(f"PS shard {shard_id} detenido. "
                f"Updates procesados: {shard.num_updates}")


class ParameterServerCoordinator:
    """
    Coordinador del Parameter Server.
    
    Soporta lanzar todos los shards en local, o un subconjunto de shards
    distribuidos en múltiples máquinas.
    """
    
    def __init__(self, config: ParameterServerConfig, total_params: int,
                 distributed: bool = False, host: str = "0.0.0.0", base_port: int = 29500,
                 shards_to_launch: Optional[List[int]] = None):
        """
        Args:
            config: Configuración del PS
            total_params: Número total de parámetros
            distributed: Si usar TCP en vez de Queue
            host: Host para escuchar (modo distribuido)
            base_port: Puerto base (modo distribuido)
            shards_to_launch: Lista de shard IDs a lanzar. 
                             None = todos (modo local).
                             [0], [1], ... = solo ese shard (modo distribuido).
        """
        self.config = config
        self.total_params = total_params
        self.distributed = distributed
        self.host = host
        self.base_port = base_port
        self.shards_to_launch = shards_to_launch if shards_to_launch is not None else list(range(config.num_shards))
        
        # Colas de comunicación (una por shard total del sistema)
        self.request_queues = [Queue() for _ in range(config.num_shards)]
        self.response_queue = Queue()
        self.should_stop = mp.Value('b', False)
        self.shard_processes = []
        
        # En modo distribuido, cada shard lanzado tiene su propia cola de responses por réplica
        self.per_shard_response_queues: Dict[int, Dict[int, Queue]] = {}
        
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
            f"PS Coordinator: {config.num_shards} shards totales, "
            f"lanzando: {self.shards_to_launch}, "
            f"{total_params:,} params, "
            f"modo: {'DISTRIBUIDO' if distributed else 'LOCAL'}"
        )
    
    def start(self, initial_model_state: Optional[Dict] = None):
        """Inicia los procesos de los shards especificados en shards_to_launch."""
        initial_params = None
        if initial_model_state is not None:
            initial_params = initial_model_state.get('parameters')
        
        for shard_id in self.shards_to_launch:
            start_idx, end_idx = get_parameter_shard_bounds(
                self.total_params, self.config.num_shards, shard_id
            )
            
            shard_initial = None
            if initial_params is not None:
                shard_initial = initial_params[start_idx:end_idx]
            
            if self.distributed:
                # MODO DISTRIBUIDO: el shard escucha en TCP
                shard_port = self.base_port + shard_id
                
                # FIX: No crear listener ni colas en el proceso padre.
                # En Windows con spawn(), queue.Queue y ParameterServerListener
                # contienen locks que NO se pueden pickle.
                # El listener y las colas se crean DENTRO del proceso hijo.
                p = mp.Process(
                    target=parameter_server_shard_process,
                    args=(
                        shard_id,
                        None,  # request_queue: se crea en el hijo
                        None,  # response_queues: se crea en el hijo
                        self.host,     # listener_host: el hijo crea el listener
                        shard_port,    # listener_port
                        self.should_stop,
                        self.config_dict,
                        self.total_params,
                        shard_initial
                    )
                )
                p.start()
                self.shard_processes.append(p)
                
                logger.info(
                    f"Shard {shard_id} en TCP {self.host}:{shard_port} "
                    f"(params [{start_idx}:{end_idx}])"
                )
            else:
                # MODO LOCAL: Queue compartida
                p = mp.Process(
                    target=parameter_server_shard_process,
                    args=(
                        shard_id,
                        self.request_queues[shard_id],
                        None,  # response_queues no usado en modo local directo
                        None,  # No TCP listener
                        None,
                        self.should_stop,
                        self.config_dict,
                        self.total_params,
                        shard_initial
                    )
                )
                p.start()
                self.shard_processes.append(p)
                
                logger.info(
                    f"Shard {shard_id} en Queue local "
                    f"(params [{start_idx}:{end_idx}])"
                )
        
        num_launched = len(self.shards_to_launch)
        logger.info(f"Shards iniciados: {self.shards_to_launch} ({num_launched}/{self.config.num_shards})")
        
        # En modo local, agregar un thread para despachar responses
        if not self.distributed:
            self._response_dispatcher = threading.Thread(
                target=self._dispatch_responses,
                daemon=True
            )
            self._response_dispatcher.start()
    
    def _dispatch_responses(self):
        """Thread que lee responses de cada shard y las centraliza."""
        # En modo local, necesitamos leer de las colas de cada shard
        # y depositar en response_queue central
        # Por ahora, en modo local las réplicas leen directamente
        pass
    
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
        """Retorna la cola de requests para un shard (modo local)."""
        return self.request_queues[shard_id]
    
    def get_response_queue(self) -> Queue:
        """Retorna la cola de responses central (modo local)."""
        return self.response_queue
    
    def get_shard_endpoints(self) -> List[Tuple[str, int]]:
        """Retorna los endpoints TCP para cada shard (modo distribuido)."""
        if self.distributed:
            return [(self.host, self.base_port + i) for i in range(self.config.num_shards)]
        return []
    
    def save_checkpoint(self):
        """Solicita a todos los shards que guarden su estado."""
        logger.info("Guardando checkpoint...")
        for shard_id in range(self.config.num_shards):
            msg = Message(
                msg_type=MessageType.SAVE_CHECKPOINT,
                sender_id=-1,
                data={}
            )
            self.request_queues[shard_id].put(msg)
        
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
        
        acks = 0
        while acks < self.config.num_shards:
            try:
                response = self.response_queue.get(block=True, timeout=5.0)
                if response.msg_type == MessageType.CHECKPOINT_DONE:
                    acks += 1
            except queue.Empty:
                break
        
        logger.info(f"Checkpoint cargado ({acks}/{self.config.num_shards} shards)")
