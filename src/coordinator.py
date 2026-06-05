"""
Coordinador Principal del Sistema DistBelief.

Soporta dos modos de operación:
- LOCAL: PS + réplicas en la misma máquina (multiprocessing)
- DISTRIBUIDO: PS en una máquina, réplicas en otras (TCP/IP)

En modo distribuido, cada nodo del cluster ejecuta una instancia del
coordinador con un rol diferente:
  --role ps        : Solo Parameter Server
  --role replica   : Solo Model Replicas
  --role coordinator: PS + réplicas (modo completo, default)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import queue
import time
import os
import json
import logging
from datetime import datetime

from utils import (
    Message, MessageType,
    count_parameters, get_model_layers_info,
    get_parameter_shard_bounds
)
from config import (
    DistBeliefConfig, ParameterServerConfig,
    ModelReplicaConfig, NetworkConfig, DataConfig, TrainingConfig
)
from parameter_server import ParameterServerCoordinator
from model_replica import model_replica_process, create_data_splits, ReplicaTransport
from network import CIFAR10Net, LeNetCIFAR10, create_model, get_model_info


logger = logging.getLogger("Coordinator")


def _setup_logging(log_dir: str = "./logs", log_file: str = "distbelief_training.log"):
    """Configura logging global: consola + archivo para auditoría."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Evitar duplicar handlers
    has_file = any(
        isinstance(h, logging.FileHandler) and
        getattr(h, 'baseFilename', '') == os.path.abspath(log_path)
        for h in root_logger.handlers
    )

    if not has_file:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Variables de entorno para procesos hijos
    os.environ["DISTBELIEF_LOG_DIR"] = os.path.abspath(log_dir)
    os.environ["DISTBELIEF_LOG_FILE"] = log_file

    return log_path


class DistBeliefCoordinator:
    """Coordinador principal del sistema DistBelief."""

    def __init__(self, config: DistBeliefConfig,
                 role: str = "coordinator",
                 ps_host: Optional[str] = None,
                 ps_port_base: int = 29500,
                 shard_id: Optional[int] = None,
                 shard_endpoints: Optional[List[Tuple[str, int]]] = None):
        """
        Args:
            config: Configuración completa
            role: 'coordinator' (todo en uno), 'ps' (solo PS), 'replica' (solo réplicas)
            ps_host: IP del PS (requerido si role='replica')
            ps_port_base: Puerto base del PS
            shard_id: Si role='ps', ID del shard específico a lanzar.
                      None = lanzar todos los shards (local).
                      0, 1, ... = lanzar solo ese shard (distribuido).
            shard_endpoints: Lista de (host, port) para cada shard.
                            Si se proporciona, sobrescribe ps_host/ps_port_base.
                            Requerido para PS sharded distribuido.
        """
        self.config = config
        self.role = role
        self.ps_host = ps_host
        self.ps_port_base = ps_port_base
        self.shard_id = shard_id
        self.shard_endpoints = shard_endpoints

        config.validate()

        # Determinar modo
        self.distributed = (role == "replica" and ps_host is not None) or \
                           (role == "ps" and shard_id is not None)

        # Crear modelo para contar parámetros
        self.model = create_model(config.network, model_type="cifar10net")
        self.total_params = count_parameters(self.model)

        logger.info("=" * 60)
        logger.info("DistBelief Coordinator")
        logger.info("=" * 60)
        logger.info(f"Rol: {role}")
        if role == "ps" and shard_id is not None:
            logger.info(f"Shard ID: {shard_id} (solo este shard)")
        logger.info(f"Modo: {'DISTRIBUIDO' if self.distributed else 'LOCAL'}")
        if self.distributed and ps_host:
            logger.info(f"PS host: {ps_host}:{ps_port_base}")
        logger.info(f"Modelo: CIFAR10Net ({self.total_params:,} params)")
        logger.info(f"Shards PS: {config.ps.num_shards}")
        logger.info(f"Réplicas: {config.replica.num_replicas}")
        logger.info(f"Adagrad: {config.ps.use_adagrad}")
        logger.info("=" * 60)

        # Variables compartidas
        self.should_stop = Value('b', False)
        self.warm_start_done = Value('b', False)
        self.global_step = Value('i', 0)

        # Componentes
        self.ps_coordinator = None
        self.replica_processes = []
        self.start_time = None
        self.metrics_history = []

    def initialize(self):
        """Inicializa componentes según el rol."""
        log_dir = os.path.join(self.config.training.output_dir, "logs")
        log_path = _setup_logging(log_dir=log_dir)
        logger.info(f"Log de auditoría: {log_path}")

        if self.role in ("coordinator", "ps"):
            logger.info("Inicializando Parameter Server...")

            # Si se especifica shard_id, lanzar SOLO ese shard (modo distribuido)
            # Si no, lanzar todos los shards (modo local/coordinator)
            shards_to_launch = [self.shard_id] if self.shard_id is not None else list(range(self.config.ps.num_shards))

            self.ps_coordinator = ParameterServerCoordinator(
                config=self.config.ps,
                total_params=self.total_params,
                distributed=self.distributed,
                host="0.0.0.0" if self.distributed else "127.0.0.1",
                base_port=self.ps_port_base,
                shards_to_launch=shards_to_launch
            )

            initial_params = self.model.get_flat_parameters()
            self.ps_coordinator.start(initial_model_state={'parameters': initial_params})
            time.sleep(2.0)
            logger.info("PS inicializado")

        if self.role == "replica":
            logger.info("Modo réplica: conectando al PS remoto...")
            time.sleep(1.0)

    def start_training(self):
        """Inicia entrenamiento según el rol."""
        if self.role == "ps":
            if self.shard_id is not None:
                logger.info(f"Rol PS (shard {self.shard_id}): esperando conexiones de réplicas...")
                logger.info(f"Shard {self.shard_id} escuchando en puerto {self.ps_port_base + self.shard_id}")
            else:
                logger.info("Rol PS (todos los shards): esperando conexiones de réplicas...")
                logger.info(f"PS escuchando en puertos {self.ps_port_base}-{self.ps_port_base + self.config.ps.num_shards - 1}")
            # En modo PS puro, el proceso se mantiene vivo
            try:
                while not self.should_stop.value:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
            return

        logger.info("=" * 60)
        logger.info("Iniciando entrenamiento")
        logger.info("=" * 60)

        self.start_time = time.time()

        # Preparar splits de datos
        num_train = 50000
        data_splits = create_data_splits(
            num_examples=num_train,
            num_replicas=self.config.replica.num_replicas,
            seed=self.config.training.seed
        )

        logger.info(f"Datos divididos: {[len(s) for s in data_splits]} ejemplos/replica")

        # Configuraciones serializadas
        ps_config_dict = {
            'num_shards': self.config.ps.num_shards,
            'base_port': self.config.ps.base_port,
            'checkpoint_dir': self.config.ps.checkpoint_dir,
            'checkpoint_frequency': self.config.ps.checkpoint_frequency,
            'adagrad_gamma': self.config.ps.adagrad_gamma,
            'adagrad_epsilon': self.config.ps.adagrad_epsilon,
            'use_adagrad': self.config.ps.use_adagrad,
            'base_learning_rate': self.config.ps.base_learning_rate,
        }

        replica_config_dict = {
            'num_replicas': self.config.replica.num_replicas,
            'batch_size': self.config.replica.batch_size,
            'fetch_frequency': self.config.replica.fetch_frequency,
            'push_frequency': self.config.replica.push_frequency,
            'warm_start_steps': self.config.replica.warm_start_steps,
            'device': self.config.replica.device,
            'log_frequency': self.config.replica.log_frequency,
            'gradient_clipping': self.config.replica.gradient_clipping,
            'max_gradient_norm': self.config.replica.max_gradient_norm,
        }

        # En modo distribuido, calcular endpoints del PS
        shard_endpoints = None
        if self.distributed:
            if self.shard_endpoints is not None:
                # Usar endpoints explícitos (PS sharded en múltiples máquinas)
                shard_endpoints = self.shard_endpoints
            elif self.ps_host is not None:
                # Todos los shards en la misma máquina (puertos consecutivos)
                shard_endpoints = [
                    (self.ps_host, self.ps_port_base + i)
                    for i in range(self.config.ps.num_shards)
                ]
            logger.info(f"Endpoints del PS: {shard_endpoints}")

        # FASE 1: Warm Start
        logger.info("-" * 60)
        logger.info("FASE 1: Warm Start (1 replica)")
        logger.info("-" * 60)

        if self.role in ("coordinator", "replica"):
            warm_start_process = mp.Process(
                target=model_replica_process,
                args=(
                    0,                          # replica_id
                    self.ps_coordinator.request_queues if self.ps_coordinator else None,
                    self.ps_coordinator.response_queue if self.ps_coordinator else None,
                    self.should_stop,
                    self.warm_start_done,
                    self.global_step,
                    replica_config_dict,
                    ps_config_dict,
                    self.total_params,
                    data_splits[0],
                    self.config.data.data_dir,
                    True,                       # is_warm_start
                    shard_endpoints             # None en local, endpoints en distribuido
                )
            )
            warm_start_process.start()
            self.replica_processes.append(warm_start_process)

            logger.info(f"Esperando warm start ({self.config.replica.warm_start_steps} steps)...")
            while not self.warm_start_done.value:
                time.sleep(0.5)
                if not warm_start_process.is_alive():
                    logger.error("Warm start terminó inesperadamente")
                    return

            logger.info(f"Warm start completado en {time.time() - self.start_time:.1f}s")

        # FASE 2: Activar réplicas restantes
        if self.config.replica.num_replicas > 1:
            logger.info("-" * 60)
            logger.info(f"FASE 2: Activando {self.config.replica.num_replicas - 1} réplicas")
            logger.info("-" * 60)

            for replica_id in range(1, self.config.replica.num_replicas):
                p = mp.Process(
                    target=model_replica_process,
                    args=(
                        replica_id,
                        self.ps_coordinator.request_queues if self.ps_coordinator else None,
                        self.ps_coordinator.response_queue if self.ps_coordinator else None,
                        self.should_stop,
                        self.warm_start_done,
                        self.global_step,
                        replica_config_dict,
                        ps_config_dict,
                        self.total_params,
                        data_splits[replica_id],
                        self.config.data.data_dir,
                        False,
                        shard_endpoints
                    )
                )
                p.start()
                self.replica_processes.append(p)
                logger.info(f"Réplica {replica_id} iniciada")

    def monitor_training(self):
        """Monitorea el entrenamiento."""
        logger.info("Monitoreando entrenamiento...")

        last_step = 0
        last_time = time.time()

        while not self.should_stop.value:
            time.sleep(5.0)

            current_step = self.global_step.value
            ws_done = self.warm_start_done.value
            ws_status = "COMPLETADO" if ws_done else "EN_PROGRESO"

            current_time = time.time()
            elapsed = current_time - last_time
            steps_per_sec = (current_step - last_step) / elapsed if elapsed > 0 else 0
            last_step = current_step
            last_time = current_time

            logger.info(
                f"Step {current_step} | "
                f"Throughput: {steps_per_sec:.1f} steps/s | "
                f"Warm: {ws_status}"
            )

            self.metrics_history.append({
                'timestamp': current_time - self.start_time,
                'global_step': current_step,
                'steps_per_sec': steps_per_sec,
            })

            if current_step > 0 and current_step % self.config.ps.checkpoint_frequency == 0:
                if self.ps_coordinator:
                    self.ps_coordinator.save_checkpoint()

            if self.config.training.max_steps > 0 and \
                    current_step >= self.config.training.max_steps:
                logger.info(f"Máximo de steps alcanzado ({current_step})")
                break

            all_dead = all(not p.is_alive() for p in self.replica_processes)
            if all_dead and current_step > self.config.replica.warm_start_steps:
                logger.info("Todas las réplicas terminaron")
                break

    def evaluate(self) -> Dict[str, float]:
        """Evalúa el modelo en test set."""
        logger.info("Evaluando modelo...")

        self._sync_parameters_from_ps()

        device = torch.device(self.config.replica.device)
        self.model = self.model.to(device)

        from torchvision import datasets, transforms

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            ),
        ])

        test_dataset = datasets.CIFAR10(
            root=self.config.data.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.replica.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(test_loader)

        logger.info(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def _sync_parameters_from_ps(self):
        """Sincroniza parámetros del PS al modelo local."""
        logger.info("Sincronizando parámetros desde PS...")

        if self.ps_coordinator is None:
            logger.warning("No hay PS local para sincronizar")
            return

        for shard_id in range(self.config.ps.num_shards):
            msg = Message(
                msg_type=MessageType.GET_PARAMETERS,
                sender_id=-1,
                data={}
            )
            self.ps_coordinator.request_queues[shard_id].put(msg)

        params = torch.zeros(self.total_params)
        received = 0

        timeout = time.time() + 10.0
        while received < self.config.ps.num_shards and time.time() < timeout:
            try:
                response = self.ps_coordinator.response_queue.get(
                    block=True, timeout=0.5
                )
                if response.msg_type == MessageType.PARAMETERS_RESPONSE:
                    start_idx = response.data['start_idx']
                    end_idx = response.data['end_idx']
                    shard_params = response.data['parameters']
                    params[start_idx:end_idx] = shard_params
                    received += 1
            except queue.Empty:
                continue

        self.model.set_flat_parameters(params)
        logger.info(f"Params sincronizados ({received}/{self.config.ps.num_shards})")

    def save_final_model(self, filepath: str = None):
        """Guarda modelo final."""
        if filepath is None:
            filepath = os.path.join(self.config.training.output_dir, "final_model.pth")
        self._sync_parameters_from_ps()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history
        }, filepath)
        logger.info(f"Modelo guardado en {filepath}")

    def save_metrics(self, filepath: str = None):
        """Guarda métricas."""
        if filepath is None:
            filepath = os.path.join(self.config.training.output_dir, "metrics.json")
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Métricas guardadas en {filepath}")

    def shutdown(self):
        """Detiene todos los procesos."""
        logger.info("=" * 60)
        logger.info("Shutting down DistBelief")
        logger.info("=" * 60)

        self.should_stop.value = True

        for i, p in enumerate(self.replica_processes):
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

        if self.ps_coordinator:
            self.ps_coordinator.stop()

        if self.start_time:
            logger.info(f"Tiempo total: {time.time() - self.start_time:.1f}s")

        logger.info("Shutdown completo")

    def run_full_training(self):
        """Ejecuta pipeline completo de entrenamiento."""
        try:
            self.initialize()
            self.start_training()

            if self.role in ("coordinator", "replica"):
                self.monitor_training()
                self.evaluate()
                self.save_final_model()
                self.save_metrics()

            return {'success': True}

        except KeyboardInterrupt:
            logger.info("Interrumpido por el usuario")
            return {'success': False, 'reason': 'interrupted'}

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'reason': str(e)}

        finally:
            self.shutdown()
