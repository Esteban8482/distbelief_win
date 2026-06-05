"""
Coordinador Principal del Sistema DistBelief.

Orquesta el Parameter Server sharded y las Model Replicas,
implementando el flujo completo de entrenamiento incluyendo:
- Inicialización del PS y réplicas
- Warm start (entrenar con 1 replica antes de activar las demás)
- Checkpointing periódico
- Evaluación y logging de métricas
- Shutdown graceful

Basado en la arquitectura descrita en Sección 4 del paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
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
from model_replica import model_replica_process, create_data_splits
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
    
    # Configurar root logger para capturar todos los logs (Coordinator, ParameterServer, ModelReplica, etc.)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Evitar duplicar handlers si ya existen
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path) for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Variables de entorno para que los procesos hijos sepan dónde escribir
    os.environ["DISTBELIEF_LOG_DIR"] = os.path.abspath(log_dir)
    os.environ["DISTBELIEF_LOG_FILE"] = log_file
    
    return log_path


class DistBeliefCoordinator:
    """
    Coordinador principal del sistema DistBelief.
    
    Gestiona el ciclo de vida completo del entrenamiento distribuido:
    1. Inicializar Parameter Server con sharding
    2. Inicializar Model Replicas con partición de datos
    3. Ejecutar warm start (1 replica primero)
    4. Activar réplicas restantes
    5. Monitorear entrenamiento y hacer checkpointing
    """
    
    def __init__(self, config: DistBeliefConfig):
        """
        Args:
            config: Configuración completa del sistema
        """
        self.config = config
        config.validate()
        
        # Crear modelo dummy para obtener info de parámetros
        self.model = create_model(config.network, model_type="cifar10net")
        self.total_params = count_parameters(self.model)
        
        logger.info("=" * 60)
        logger.info("DistBelief Coordinator")
        logger.info("=" * 60)
        logger.info(f"Modelo: CIFAR10Net")
        logger.info(f"Parámetros totales: {self.total_params:,}")
        logger.info(f"Shards del PS: {config.ps.num_shards}")
        logger.info(f"Model replicas: {config.replica.num_replicas}")
        logger.info(f"Warm start steps: {config.replica.warm_start_steps}")
        logger.info(f"Usando Adagrad: {config.ps.use_adagrad}")
        if config.ps.use_adagrad:
            logger.info(f"  Adagrad gamma: {config.ps.adagrad_gamma}")
        logger.info("=" * 60)
        
        # Variables compartidas entre procesos
        self.should_stop = Value('b', False)
        self.warm_start_done = Value('b', False)
        self.global_step = Value('i', 0)
        
        # Componentes
        self.ps_coordinator = None
        self.replica_processes = []
        
        # Métricas
        self.start_time = None
        self.metrics_history = []
    
    def initialize(self):
        """Inicializa todos los componentes del sistema."""
        # Configurar logging a archivo para auditoría del entrenamiento
        log_dir = os.path.join(self.config.training.output_dir, "logs")
        log_path = _setup_logging(log_dir=log_dir, log_file="distbelief_training.log")
        logger.info(f"Archivo de log de auditoría: {log_path}")
        
        logger.info("Inicializando sistema DistBelief...")
        
        # Inicializar Parameter Server
        self.ps_coordinator = ParameterServerCoordinator(
            config=self.config.ps,
            total_params=self.total_params
        )
        
        # Obtener parámetros iniciales del modelo
        initial_params = self.model.get_flat_parameters()
        initial_state = {'parameters': initial_params}
        
        # Iniciar PS shards
        self.ps_coordinator.start(initial_model_state=initial_state)
        
        # Dar tiempo a que los shards inicien
        time.sleep(2.0)
        
        logger.info("Sistema inicializado correctamente")
    
    def start_training(self):
        """
        Inicia el entrenamiento con warm start.
        
        Flujo (basado en Sección 4.1 del paper):
        1. "warm starting model training with only a single model replica 
           before unleashing the other replicas"
        2. Luego activar todas las réplicas
        """
        logger.info("=" * 60)
        logger.info("Iniciando entrenamiento")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Preparar splits de datos
        # CIFAR-10 train: 50,000 ejemplos
        num_train = 50000
        data_splits = create_data_splits(
            num_examples=num_train,
            num_replicas=self.config.replica.num_replicas,
            seed=self.config.training.seed
        )
        
        logger.info(
            f"Datos divididos: "
            f"{[len(s) for s in data_splits]} ejemplos por replica"
        )
        
        # Serializar configuraciones como dicts
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
        
        # FASE 1: Warm Start
        # "combined with a practice of 'warmstarting' model training 
        # with only a single model replica before unleashing the other 
        # replicas, it has virtually eliminated stability concerns"
        logger.info("-" * 60)
        logger.info("FASE 1: Warm Start (1 replica)")
        logger.info("-" * 60)
        
        warm_start_process = mp.Process(
            target=model_replica_process,
            args=(
                0,                          # replica_id
                self.ps_coordinator.request_queues,  # ps_request_queues
                self.ps_coordinator.response_queue,  # ps_response_queue
                self.should_stop,           # should_stop
                self.warm_start_done,       # warm_start_done
                self.global_step,           # global_step
                replica_config_dict,        # replica_config
                ps_config_dict,             # ps_config
                self.total_params,          # total_params
                data_splits[0],             # Datos para warm start
                self.config.data.data_dir,  # data_dir
                True                        # is_warm_start
            )
        )
        warm_start_process.start()
        self.replica_processes.append(warm_start_process)
        
        # Esperar a que el warm start complete
        logger.info(
            f"Esperando warm start ({self.config.replica.warm_start_steps} steps)..."
        )
        while not self.warm_start_done.value:
            time.sleep(0.5)
            if not warm_start_process.is_alive():
                logger.error("El proceso de warm start terminó inesperadamente")
                return
        
        warm_start_time = time.time() - self.start_time
        logger.info(f"Warm start completado en {warm_start_time:.1f}s")
        
        # FASE 2: Activar réplicas restantes
        if self.config.replica.num_replicas > 1:
            logger.info("-" * 60)
            logger.info(f"FASE 2: Activando {self.config.replica.num_replicas - 1} réplicas adicionales")
            logger.info("-" * 60)
            
            for replica_id in range(1, self.config.replica.num_replicas):
                p = mp.Process(
                    target=model_replica_process,
                    args=(
                        replica_id,
                        self.ps_coordinator.request_queues,
                        self.ps_coordinator.response_queue,
                        self.should_stop,
                        self.warm_start_done,
                        self.global_step,
                        replica_config_dict,
                        ps_config_dict,
                        self.total_params,
                        data_splits[replica_id],
                        self.config.data.data_dir,
                        False  # No es warm start
                    )
                )
                p.start()
                self.replica_processes.append(p)
                logger.info(f"Réplica {replica_id} iniciada")
        
        logger.info(
            f"Todas las {self.config.replica.num_replicas} réplicas activas"
        )
    
    def monitor_training(self):
        """
        Monitorea el entrenamiento y recolecta métricas.
        
        Corre hasta que se complete el entrenamiento o se solicite detener.
        """
        logger.info("Monitoreando entrenamiento...")
        
        last_step = 0
        last_time = time.time()
        start_monitor_time = time.time()
        
        max_monitor_time = None
        if self.config.training.num_epochs > 0:
            # Estimar tiempo máximo (aproximado)
            max_monitor_time = self.config.training.num_epochs * 600  # ~10 min por epoch
        
        while not self.should_stop.value:
            time.sleep(5.0)  # Log cada 5 segundos
            
            # Obtener estado actual
            current_step = self.global_step.value
            
            # Calcular throughput
            current_time = time.time()
            elapsed = current_time - last_time
            steps_per_sec = (current_step - last_step) / elapsed if elapsed > 0 else 0
            last_step = current_step
            last_time = current_time
            
            # Log (FIX: leer warm_start_done.value explícitamente cada iter)
            ws_done = self.warm_start_done.value
            ws_status = "COMPLETADO" if ws_done else "EN_PROGRESO"
            logger.info(
                f"Step {current_step} | "
                f"Throughput: {steps_per_sec:.1f} steps/s | "
                f"Warm: {ws_status}"
            )
            
            # Guardar métricas
            self.metrics_history.append({
                'timestamp': current_time - self.start_time,
                'global_step': current_step,
                'steps_per_sec': steps_per_sec,
            })
            
            # Checkpoint periódico
            if current_step > 0 and current_step % self.config.ps.checkpoint_frequency == 0:
                self.ps_coordinator.save_checkpoint()
            
            # Verificar si se alcanzó el máximo de steps
            if (self.config.training.max_steps > 0 and 
                current_step >= self.config.training.max_steps):
                logger.info(f"Máximo de steps alcanzado ({current_step})")
                break
            
            # Verificar si todas las replicas terminaron
            all_dead = all(
                not p.is_alive() for p in self.replica_processes
            )
            if all_dead and current_step > self.config.replica.warm_start_steps:
                logger.info("Todas las réplicas terminaron")
                break
            
            # Timeout de seguridad
            if max_monitor_time and (current_time - start_monitor_time) > max_monitor_time:
                logger.info("Timeout de monitoreo alcanzado")
                break
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de test.
        
        Returns:
            Dict con métricas de evaluación
        """
        logger.info("Evaluando modelo...")
        
        # Obtener parámetros actuales del PS
        self._sync_parameters_from_ps()
        
        device = torch.device(self.config.replica.device)
        self.model = self.model.to(device)
        
        # Cargar dataset de test
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
        """Sincroniza los parámetros del PS al modelo local."""
        logger.info("Sincronizando parámetros desde PS...")
        
        # Solicitar parámetros de todos los shards
        for shard_id in range(self.config.ps.num_shards):
            msg = Message(
                msg_type=MessageType.GET_PARAMETERS,
                sender_id=-1,
                data={}
            )
            self.ps_coordinator.request_queues[shard_id].put(msg)
        
        # Recolectar respuestas
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
        logger.info(f"Parámetros sincronizados ({received}/{self.config.ps.num_shards} shards)")
    
    def save_final_model(self, filepath: str = None):
        """Guarda el modelo final entrenado."""
        if filepath is None:
            filepath = os.path.join(
                self.config.training.output_dir,
                "final_model.pth"
            )
        
        # Sincronizar parámetros
        self._sync_parameters_from_ps()
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history
        }, filepath)
        
        logger.info(f"Modelo guardado en {filepath}")
    
    def save_metrics(self, filepath: str = None):
        """Guarda las métricas de entrenamiento."""
        if filepath is None:
            filepath = os.path.join(
                self.config.training.output_dir,
                "metrics.json"
            )
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Métricas guardadas en {filepath}")
    
    def shutdown(self):
        """Detiene todos los procesos de forma graceful."""
        logger.info("=" * 60)
        logger.info("Shutting down DistBelief")
        logger.info("=" * 60)
        
        # Señalizar parada
        self.should_stop.value = True
        
        # Esperar a que las replicas terminen
        logger.info("Esperando réplicas...")
        for i, p in enumerate(self.replica_processes):
            p.join(timeout=10.0)
            if p.is_alive():
                logger.warning(f"Réplica {i} no respondió, forzando terminación")
                p.terminate()
        
        # Detener PS
        if self.ps_coordinator:
            self.ps_coordinator.stop()
        
        # Calcular tiempo total
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Tiempo total de entrenamiento: {total_time:.1f}s")
        
        logger.info("Shutdown completo")
    
    def run_full_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict con resultados
        """
        try:
            # 1. Inicializar
            self.initialize()
            
            # 2. Entrenar
            self.start_training()
            
            # 3. Monitorear
            self.monitor_training()
            
            # 4. Evaluar
            eval_results = self.evaluate()
            
            # 5. Guardar resultados
            self.save_final_model()
            self.save_metrics()
            
            return {
                'success': True,
                'eval_results': eval_results,
                'metrics_history': self.metrics_history
            }
            
        except KeyboardInterrupt:
            logger.info("Entrenamiento interrumpido por el usuario")
            return {'success': False, 'reason': 'interrupted'}
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'reason': str(e)}
            
        finally:
            # Siempre hacer shutdown
            self.shutdown()
