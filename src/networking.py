"""
Capa de transporte TCP/IP para DistBelief distribuido.

Reemplaza multiprocessing.Queue con sockets TCP para permitir
comunicación entre máquinas en una red local (cluster/laboratorio).

Arquitectura:
- Parameter Server shards escuchan en puertos TCP
- Model Replicas se conectan vía TCP a los shards
- Mensajes serializados con pickle sobre sockets
- Soporte para modo local (loopback) y modo cluster (LAN)

Mantenemos la misma semántica de Message que en utils.py,
pero ahora los canales son sockets en vez de Queue.
"""

import socket
import pickle
import struct
import threading
import queue
import logging
import time
from typing import Optional, Dict, List, Tuple, Callable
from contextlib import contextmanager

from utils import Message, MessageType


logger = logging.getLogger("Networking")


# ============================================================
# SERIALIZACIÓN / DESERIALIZACIÓN DE MENSAJES SOBRE SOCKETS
# ============================================================

def _send_message(sock: socket.socket, msg: Message) -> bool:
    """
    Serializa y envía un Message a través de un socket TCP.
    
    Protocolo: [4 bytes tamaño][payload pickle]
    
    Args:
        sock: Socket conectado
        msg: Mensaje a enviar
        
    Returns:
        True si se envió correctamente
    """
    try:
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        size_header = struct.pack('!I', len(payload))
        sock.sendall(size_header + payload)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        return False
    except Exception as e:
        logger.debug(f"Error enviando mensaje: {e}")
        return False


def _recv_message(sock: socket.socket, timeout: Optional[float] = None) -> Optional[Message]:
    """
    Recibe y deserializa un Message de un socket TCP.
    
    Args:
        sock: Socket conectado
        timeout: Timeout en segundos (None = bloqueante)
        
    Returns:
        Message deserializado, o None si timeout/error
    """
    try:
        if timeout is not None:
            sock.settimeout(timeout)
        
        # Leer 4 bytes de header (tamaño)
        header = _recv_all(sock, 4)
        if header is None:
            return None
        
        payload_size = struct.unpack('!I', header)[0]
        
        # Validar tamaño (protección contra corruptos/maliciosos)
        if payload_size > 100 * 1024 * 1024:  # Max 100 MB
            logger.error(f"Payload demasiado grande: {payload_size} bytes")
            return None
        
        # Leer payload
        payload = _recv_all(sock, payload_size)
        if payload is None:
            return None
        
        msg = pickle.loads(payload)
        return msg
    
    except socket.timeout:
        return None
    except (ConnectionResetError, ConnectionAbortedError, OSError):
        return None
    except Exception as e:
        logger.debug(f"Error recibiendo mensaje: {e}")
        return None
    finally:
        sock.settimeout(None)


def _recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    """Lee exactamente n bytes del socket."""
    data = b''
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        except (ConnectionResetError, ConnectionAbortedError, OSError):
            return None
    return data


# ============================================================
# PARAMETER SERVER LISTENER (escucha conexiones TCP)
# ============================================================

class ParameterServerListener:
    """
    Listener TCP para un shard del Parameter Server.
    
    Escucha conexiones entrantes de model replicas y gestiona
    el procesamiento asíncrono de mensajes.
    
    Semántica equivalente a Queue.get() / Queue.put() pero sobre TCP.
    """
    
    def __init__(self, host: str, port: int, request_queue, response_queues: Dict[int, queue.Queue]):
        """
        Args:
            host: IP/hostname para escuchar ('0.0.0.0' para todas)
            port: Puerto TCP
            request_queue: Cola local donde depositar requests entrantes
            response_queues: Dict replica_id -> cola de responses salientes
        """
        self.host = host
        self.port = port
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.sock = None
        self.running = False
        self.listener_thread = None
        self.replica_socks: Dict[int, socket.socket] = {}
        self.lock = threading.Lock()
    
    def start(self):
        """Inicia el listener en un thread separado."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)  # Backlog de 10 conexiones
        self.running = True
        
        self.listener_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.listener_thread.start()
        
        logger.info(f"PS Listener iniciado en {self.host}:{self.port}")
    
    def stop(self):
        """Detiene el listener y cierra conexiones."""
        self.running = False
        try:
            self.sock.close()
        except:
            pass
        with self.lock:
            for sock in self.replica_socks.values():
                try:
                    sock.close()
                except:
                    pass
        logger.info(f"PS Listener detenido")
    
    def _accept_loop(self):
        """Loop principal: acepta conexiones de réplicas."""
        while self.running:
            try:
                self.sock.settimeout(1.0)
                client_sock, addr = self.sock.accept()
                self.sock.settimeout(None)
                
                logger.info(f"Réplica conectada desde {addr}")
                
                # Handshake: recibir replica_id
                client_sock.settimeout(5.0)
                handshake = _recv_message(client_sock)
                client_sock.settimeout(None)
                
                if handshake and handshake.msg_type == MessageType.START_TRAINING:
                    replica_id = handshake.data.get('replica_id', -1)
                    with self.lock:
                        self.replica_socks[replica_id] = client_sock
                    
                    # Iniciar threads para lectura y escritura
                    read_thread = threading.Thread(
                        target=self._read_from_replica,
                        args=(client_sock, replica_id),
                        daemon=True
                    )
                    read_thread.start()
                    
                    write_thread = threading.Thread(
                        target=self._write_to_replica,
                        args=(client_sock, replica_id),
                        daemon=True
                    )
                    write_thread.start()
                else:
                    client_sock.close()
            
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                logger.error(f"Error en accept: {e}")
    
    def _read_from_replica(self, sock: socket.socket, replica_id: int):
        """Lee mensajes de una réplica y los deposita en request_queue."""
        while self.running:
            msg = _recv_message(sock, timeout=1.0)
            if msg is None:
                continue
            
            # Actualizar sender_id si es necesario
            msg.sender_id = replica_id
            
            try:
                self.request_queue.put(msg, block=False)
            except queue.Full:
                logger.warning(f"Request queue llena, descartando mensaje de replica {replica_id}")
    
    def _write_to_replica(self, sock: socket.socket, replica_id: int):
        """Lee responses de la cola y los envía a la réplica."""
        q = self.response_queues.get(replica_id)
        if q is None:
            return
        
        while self.running:
            try:
                msg = q.get(block=True, timeout=0.5)
                if not _send_message(sock, msg):
                    break
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"Error escribiendo a replica {replica_id}: {e}")
                break


# ============================================================
# MODEL REPLICA CLIENT (conecta al PS vía TCP)
# ============================================================

class ModelReplicaClient:
    """
    Cliente TCP para una Model Replica.
    
    Se conecta a los shards del Parameter Server y gestiona
    el envío de requests y recepción de responses.
    
    Semántica equivalente a Queue.put() / Queue.get() pero sobre TCP.
    """
    
    def __init__(self, replica_id: int, shard_endpoints: List[Tuple[str, int]]):
        """
        Args:
            replica_id: ID de esta réplica
            shard_endpoints: Lista de (host, port) para cada shard del PS
        """
        self.replica_id = replica_id
        self.shard_endpoints = shard_endpoints
        self.socks: List[socket.socket] = []
        self.running = False
        self.response_queue = queue.Queue()
        self.read_threads = []
    
    def connect(self) -> bool:
        """Se conecta a todos los shards del PS."""
        all_connected = True
        
        for shard_id, (host, port) in enumerate(self.shard_endpoints):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((host, port))
                sock.settimeout(None)
                self.socks.append(sock)
                
                # Handshake: enviar replica_id
                handshake = Message(
                    msg_type=MessageType.START_TRAINING,
                    sender_id=self.replica_id,
                    data={'replica_id': self.replica_id}
                )
                _send_message(sock, handshake)
                
                logger.info(f"Réplica {self.replica_id} conectada a PS shard {shard_id} en {host}:{port}")
            
            except Exception as e:
                logger.error(f"Réplica {self.replica_id}: Fallo conectando a {host}:{port}: {e}")
                all_connected = False
        
        if all_connected:
            self.running = True
            for sock in self.socks:
                t = threading.Thread(target=self._read_loop, args=(sock,), daemon=True)
                t.start()
                self.read_threads.append(t)
        
        return all_connected
    
    def disconnect(self):
        """Cierra todas las conexiones."""
        self.running = False
        for sock in self.socks:
            try:
                sock.close()
            except:
                pass
        self.socks = []
        logger.info(f"Réplica {self.replica_id} desconectada del PS")
    
    def send_to_shard(self, shard_id: int, msg: Message) -> bool:
        """Envía un mensaje a un shard específico."""
        if shard_id < len(self.socks):
            return _send_message(self.socks[shard_id], msg)
        return False
    
    def receive(self, block: bool = True, timeout: float = 1.0) -> Optional[Message]:
        """Recibe un mensaje del PS (cualquier shard)."""
        try:
            return self.response_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def _read_loop(self, sock: socket.socket):
        """Loop que lee mensajes de un socket y los deposita en response_queue."""
        while self.running:
            msg = _recv_message(sock, timeout=1.0)
            if msg is not None:
                try:
                    self.response_queue.put(msg, block=False)
                except queue.Full:
                    pass


# ============================================================
# DETECCIÓN DE MODO: LOCAL vs DISTRIBUIDO
# ============================================================

def is_distributed_mode(config: dict) -> bool:
    """
    Determina si estamos en modo distribuido o local.
    
    Modo distribuido: se especifican IPs de los shards del PS.
    Modo local: todo en localhost (multiprocessing).
    
    Args:
        config: Dict con 'ps_host' o 'shard_endpoints'
        
    Returns:
        True si es modo distribuido
    """
    return config.get('ps_host') is not None or config.get('shard_endpoints') is not None


def get_shard_endpoints(config) -> List[Tuple[str, int]]:
    """
    Obtiene la lista de endpoints (host, port) para los shards del PS.
    
    Puede venir de:
    - Archivo de configuración del cluster
    - Variables de entorno
    - Argumentos de línea de comandos
    
    Returns:
        Lista de (host, port) tuples
    """
    import os
    
    # 1. Intentar leer de variable de entorno
    env_endpoints = os.environ.get('DISTBELIEF_PS_ENDPOINTS')
    if env_endpoints:
        endpoints = []
        for ep in env_endpoints.split(','):
            host, port = ep.strip().split(':')
            endpoints.append((host, int(port)))
        return endpoints
    
    # 2. Intentar leer de archivo de configuración del cluster
    cluster_file = os.environ.get('DISTBELIEF_CLUSTER_FILE')
    if cluster_file and os.path.exists(cluster_file):
        import json
        with open(cluster_file, 'r') as f:
            cluster_config = json.load(f)
        
        endpoints = []
        for shard in cluster_config.get('parameter_server_shards', []):
            endpoints.append((shard['host'], shard['port']))
        return endpoints
    
    # 3. Fallback: localhost con puertos base
    base_port = getattr(config, 'base_port', 29500) if hasattr(config, 'base_port') else 29500
    num_shards = getattr(config, 'num_shards', 1) if hasattr(config, 'num_shards') else 1
    
    return [('127.0.0.1', base_port + i) for i in range(num_shards)]


# ============================================================
# UTILIDADES DE RED
# ============================================================

def get_local_ip() -> str:
    """Obtiene la IP local de la máquina en la red LAN."""
    try:
        # Conecta a un destino externo para determinar la IP de salida
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def test_connectivity(host: str, port: int, timeout: float = 2.0) -> bool:
    """Prueba si se puede conectar a un endpoint."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False
