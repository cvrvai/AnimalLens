"""
Training module for AnimalLens custom model fine-tuning and transfer learning.
"""
from animallens.training.dataset_builder import VideoDatasetBuilder
from animallens.training.manager import EpochMetric, TrainingJob, TrainingJobManager, training_manager
from animallens.training.trainer import ModelTrainer, TrainingReport

__all__ = [
    "VideoDatasetBuilder",
    "ModelTrainer",
    "TrainingReport",
    "TrainingJob",
    "EpochMetric",
    "TrainingJobManager",
    "training_manager",
]
