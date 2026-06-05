"""
Red Neuronal para CIFAR-10 compatible con DistBelief.

Diseñada para ser particionable (model parallelism) y funcionar
con el Parameter Server sharded.

Basada en la arquitectura descrita en el paper para las tareas
de visión: capas convolucionales con conectividad local seguidas
de capas fully-connected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from config import NetworkConfig


class CIFAR10Net(nn.Module):
    """
    Red neuronal para clasificación de CIFAR-10.
    
    Arquitectura:
    - 3 capas convolucionales con BatchNorm y MaxPool
    - Capas fully-connected con Dropout
    - Softmax en la salida
    
    Diseñada para ser compatible con el sharding del Parameter Server.
    """
    
    def __init__(self, config: NetworkConfig = None):
        """
        Args:
            config: Configuración de la red
        """
        super(CIFAR10Net, self).__init__()
        
        if config is None:
            config = NetworkConfig()
        self.config = config
        
        # Capas convolucionales
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if config.use_batch_norm else None
        self.pool = nn.MaxPool2d(
            kernel_size=config.pool_size, 
            stride=config.pool_size
        )
        
        in_channels = config.input_channels
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
            config.conv_layers
        ):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            if config.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        # Calcular tamaño de la salida de las capas convolucionales
        # CIFAR-10: 32x32
        # Después de conv + pool: 32 -> 16 -> 8 -> 4
        self.feature_size = self._calculate_feature_size()
        
        # Capas fully-connected
        self.fc_layers = nn.ModuleList()
        fc_input_size = self.feature_size
        
        for i, fc_output_size in enumerate(config.fc_layers):
            self.fc_layers.append(nn.Linear(fc_input_size, fc_output_size))
            fc_input_size = fc_output_size
        
        # Capa de salida
        self.output_layer = nn.Linear(fc_input_size, config.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self._initialize_weights()
    
    def _calculate_feature_size(self) -> int:
        """Calcula el tamaño del vector de features después de las conv."""
        with torch.no_grad():
            x = torch.zeros(
                1, self.config.input_channels,
                self.config.input_height, self.config.input_width
            )
            for i, conv in enumerate(self.conv_layers):
                x = conv(x)
                if self.config.use_batch_norm:
                    x = self.bn_layers[i](x)
                x = F.relu(x)
                x = self.pool(x)
            return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Inicializa los pesos usando Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch, channels, height, width]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Capas convolucionales
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.config.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.pool(x)
        
        # Aplanar
        x = x.view(x.size(0), -1)
        
        # Capas fully-connected
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Capa de salida
        x = self.output_layer(x)
        
        return x
    
    def get_flat_parameters(self) -> torch.Tensor:
        """
        Obtiene todos los parámetros como un tensor 1D.
        
        Returns:
            Tensor 1D con todos los parámetros concatenados
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def set_flat_parameters(self, flat_params: torch.Tensor):
        """
        Establece los parámetros desde un tensor 1D.
        
        Args:
            flat_params: Tensor 1D con los parámetros
        """
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(
                flat_params[offset:offset + numel].view_as(param)
            )
            offset += numel
    
    def get_flat_gradients(self) -> torch.Tensor:
        """
        Obtiene todos los gradientes como un tensor 1D.
        
        Returns:
            Tensor 1D con todos los gradientes concatenados
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(torch.zeros(param.numel()))
        return torch.cat(grads)


class LeNetCIFAR10(nn.Module):
    """
    Variante de LeNet para CIFAR-10.
    
    Arquitectura más simple para comparación y debugging rápido.
    """
    
    def __init__(self, num_classes: int = 10):
        super(LeNetCIFAR10, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_flat_parameters(self) -> torch.Tensor:
        """Obtiene parámetros aplanados."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_parameters(self, flat_params: torch.Tensor):
        """Establece parámetros desde tensor aplanado."""
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(
                flat_params[offset:offset + numel].view_as(param)
            )
            offset += numel
    
    def get_flat_gradients(self) -> torch.Tensor:
        """Obtiene gradientes aplanados."""
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(torch.zeros(param.numel()))
        return torch.cat(grads)


def create_model(config: NetworkConfig = None, model_type: str = "cifar10net"):
    """
    Factory para crear modelos.
    
    Args:
        config: Configuración de la red
        model_type: Tipo de modelo ('cifar10net' o 'lenet')
        
    Returns:
        Modelo instanciado
    """
    if config is None:
        config = NetworkConfig()
    
    if model_type == "cifar10net":
        return CIFAR10Net(config)
    elif model_type == "lenet":
        return LeNetCIFAR10(config.num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Obtiene información resumida del modelo."""
    total_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'layers': []
    }
    
    for name, param in model.named_parameters():
        info['layers'].append({
            'name': name,
            'shape': list(param.shape),
            'num_params': param.numel()
        })
    
    return info
