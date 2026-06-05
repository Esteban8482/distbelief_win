"""
DistBelief - Implementación de Distributed Deep Learning
Basado en el paper "Large Scale Distributed Deep Networks" (Dean et al., NIPS 2012)

Este paquete implementa:
- Parameter Server con sharding
- Downpour SGD (asíncrono)
- Adagrad adaptive learning rates
- Model replicas con warm start
- Model parallelism
"""

__version__ = "1.0.0"
