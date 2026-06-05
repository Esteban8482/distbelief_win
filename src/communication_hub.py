"""
Hub de comunicación global para procesos DistBelief.

Este módulo implementa un patrón de registro global para Queue
de multiprocessing, permitiendo que procesos creados con spawn
puedan acceder a las mismas colas sin necesidad de serializarlas.

Uso:
    # Proceso principal: registrar queues
    hub.register('ps_requests_0', Queue())
    hub.register('ps_requests_1', Queue())
    hub.register('ps_responses', Queue())
    
    # Procesos hijos: acceder por nombre
    q = hub.get('ps_requests_0')
    q.put(message)
"""

import multiprocessing as mp
from multiprocessing import Queue, Value
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("CommunicationHub")


class _CommunicationHub:
    """
    Hub singleton para comunicación entre procesos.
    
    Almacena Queue y Values de multiprocessing que pueden ser
    accedidas por nombre desde cualquier proceso.
    """
    
    def __init__(self):
        self._queues: Dict[str, Queue] = {}
        self._values: Dict[str, Any] = {}
        self._initialized = False
    
    def register_queue(self, name: str, queue: Queue):
        """Registra una Queue con un nombre."""
        self._queues[name] = queue
        logger.debug(f"Queue registrada: {name}")
    
    def get_queue(self, name: str) -> Optional[Queue]:
        """Obtiene una Queue por nombre."""
        return self._queues.get(name)
    
    def register_value(self, name: str, value):
        """Registra un Value con un nombre."""
        self._values[name] = value
        logger.debug(f"Value registrado: {name}")
    
    def get_value(self, name: str):
        """Obtiene un Value por nombre."""
        return self._values.get(name)
    
    def create_ps_queues(self, num_shards: int) -> tuple:
        """
        Crea y registra todas las colas necesarias para el PS.
        
        Returns:
            (request_queues, response_queue, should_stop, warm_start_done, global_step)
        """
        # Colas de requests para cada shard
        request_queues = []
        for i in range(num_shards):
            q = Queue()
            name = f"ps_request_{i}"
            self.register_queue(name, q)
            request_queues.append(q)
        
        # Cola de responses
        response_queue = Queue()
        self.register_queue("ps_response", response_queue)
        
        # Values compartidos
        should_stop = Value('b', False)
        self.register_value("should_stop", should_stop)
        
        warm_start_done = Value('b', False)
        self.register_value("warm_start_done", warm_start_done)
        
        global_step = Value('i', 0)
        self.register_value("global_step", global_step)
        
        self._initialized = True
        
        return request_queues, response_queue, should_stop, warm_start_done, global_step
    
    def get_ps_queues(self, num_shards: int) -> tuple:
        """
        Obtiene las colas del PS por nombre.
        
        Returns:
            Lista de request queues
        """
        request_queues = []
        for i in range(num_shards):
            q = self.get_queue(f"ps_request_{i}")
            if q is None:
                raise RuntimeError(f"Queue ps_request_{i} no registrada")
            request_queues.append(q)
        
        response_queue = self.get_queue("ps_response")
        if response_queue is None:
            raise RuntimeError("Queue ps_response no registrada")
        
        return request_queues, response_queue
    
    def clear(self):
        """Limpia todos los recursos."""
        self._queues.clear()
        self._values.clear()
        self._initialized = False


# Instancia global singleton
_hub = _CommunicationHub()


def get_hub() -> _CommunicationHub:
    """Retorna la instancia global del hub."""
    return _hub


def register_queue(name: str, queue: Queue):
    """Registra una Queue globalmente."""
    _hub.register_queue(name, queue)


def get_queue(name: str) -> Optional[Queue]:
    """Obtiene una Queue globalmente."""
    return _hub.get_queue(name)


def register_value(name: str, value):
    """Registra un Value globalmente."""
    _hub.register_value(name, value)


def get_value(name: str):
    """Obtiene un Value globalmente."""
    return _hub.get_value(name)
