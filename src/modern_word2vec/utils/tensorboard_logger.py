"""TensorBoard logging utilities for Word2Vec training."""

import os
import time
from typing import Dict, Optional, Any
import psutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Comprehensive TensorBoard logger for Word2Vec training."""

    def __init__(
        self,
        log_dir: str,
        log_gradients: bool = False,
        log_weights: bool = False,
        log_system_stats: bool = False,
        experiment_name: Optional[str] = None,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            log_gradients: Whether to log gradient statistics
            log_weights: Whether to log model weights
            log_system_stats: Whether to log system statistics
            experiment_name: Optional experiment name for subdirectory
        """
        self.enable_gradient_logging = log_gradients
        self.enable_weight_logging = log_weights
        self.enable_system_logging = log_system_stats

        # Create experiment-specific directory
        if experiment_name:
            log_dir = os.path.join(log_dir, experiment_name)
        else:
            # Use timestamp for unique run identification
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(log_dir, f"run_{timestamp}")

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

        # Initialize system monitoring
        if self.enable_system_logging:
            self.process = psutil.Process()
            self.initial_memory = psutil.virtual_memory().used

        print(f"TensorBoard logging to: {log_dir}")
        print(f"  View with: tensorboard --logdir {log_dir}")

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metrics
        """
        # Convert non-serializable values to strings
        serializable_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                serializable_hparams[key] = value
            else:
                serializable_hparams[key] = str(value)

        self.writer.add_hparams(serializable_hparams, metrics)

    def log_training_metrics(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        epoch: int,
    ):
        """Log basic training metrics.

        Args:
            step: Global training step
            loss: Current loss value
            learning_rate: Current learning rate
            batch_size: Batch size
            epoch: Current epoch
        """
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/learning_rate", learning_rate, step)
        self.writer.add_scalar("train/batch_size", batch_size, step)
        self.writer.add_scalar("train/epoch", epoch, step)

    def log_gradient_stats(self, model: nn.Module, step: int):
        """Log gradient statistics.

        Args:
            model: PyTorch model
            step: Global training step
        """
        if not self.enable_gradient_logging:
            return

        total_norm = 0.0
        param_count = 0
        gradient_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Log individual parameter gradient norms
                self.writer.add_scalar(f"gradients/norm/{name}", param_norm, step)

                # Log gradient statistics
                grad_data = param.grad.data
                self.writer.add_scalar(f"gradients/mean/{name}", grad_data.mean(), step)
                self.writer.add_scalar(f"gradients/std/{name}", grad_data.std(), step)
                self.writer.add_scalar(f"gradients/max/{name}", grad_data.max(), step)
                self.writer.add_scalar(f"gradients/min/{name}", grad_data.min(), step)

                # Store for histogram
                gradient_norms[name] = param_norm.item()

        # Log total gradient norm
        total_norm = total_norm ** (1.0 / 2)
        self.writer.add_scalar("gradients/total_norm", total_norm, step)

        # Log gradient norm histogram
        if gradient_norms:
            self.writer.add_histogram(
                "gradients/norm_distribution",
                torch.tensor(list(gradient_norms.values())),
                step,
            )

    def log_weight_stats(self, model: nn.Module, step: int):
        """Log model weight statistics.

        Args:
            model: PyTorch model
            step: Global training step
        """
        if not self.enable_weight_logging:
            return

        for name, param in model.named_parameters():
            if param.data is not None:
                # Log weight statistics
                weight_data = param.data
                self.writer.add_scalar(f"weights/mean/{name}", weight_data.mean(), step)
                self.writer.add_scalar(f"weights/std/{name}", weight_data.std(), step)
                self.writer.add_scalar(f"weights/max/{name}", weight_data.max(), step)
                self.writer.add_scalar(f"weights/min/{name}", weight_data.min(), step)
                self.writer.add_scalar(f"weights/norm/{name}", weight_data.norm(), step)

                # Log weight histograms (less frequently to save space)
                if step % 1000 == 0:
                    self.writer.add_histogram(f"weights/{name}", weight_data, step)

    def log_system_stats(self, step: int, device: torch.device):
        """Log system statistics.

        Args:
            step: Global training step
            device: Training device
        """
        if not self.enable_system_logging:
            return

        # CPU statistics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.writer.add_scalar("system/cpu_percent", cpu_percent, step)
        self.writer.add_scalar("system/cpu_count", psutil.cpu_count(), step)

        # Memory statistics
        memory = psutil.virtual_memory()
        self.writer.add_scalar("system/memory_percent", memory.percent, step)
        self.writer.add_scalar("system/memory_used_gb", memory.used / 1e9, step)
        self.writer.add_scalar("system/memory_available_gb", memory.available / 1e9, step)

        # Process-specific memory
        process_memory = self.process.memory_info()
        self.writer.add_scalar("system/process_memory_mb", process_memory.rss / 1e6, step)

        # GPU statistics (if available)
        if device.type == "cuda" and torch.cuda.is_available():
            gpu_id = device.index or 0
            
            # GPU memory
            gpu_memory = torch.cuda.memory_stats(device)
            allocated_mb = gpu_memory.get("allocated_bytes.all.current", 0) / 1e6
            reserved_mb = gpu_memory.get("reserved_bytes.all.current", 0) / 1e6
            
            self.writer.add_scalar("system/gpu_memory_allocated_mb", allocated_mb, step)
            self.writer.add_scalar("system/gpu_memory_reserved_mb", reserved_mb, step)
            
            # GPU utilization (if nvidia-ml-py is available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                self.writer.add_scalar("system/gpu_utilization_percent", gpu_util.gpu, step)
                self.writer.add_scalar("system/gpu_memory_utilization_percent", gpu_util.memory, step)
                self.writer.add_scalar("system/gpu_temperature_c", gpu_temp, step)
            except ImportError:
                pass  # pynvml not available
            except Exception:
                pass  # Other NVIDIA ML errors

        elif device.type == "mps":
            # MPS (Apple Silicon) statistics
            if hasattr(torch.backends.mps, "current_allocated_memory"):
                mps_allocated = torch.backends.mps.current_allocated_memory() / 1e6
                self.writer.add_scalar("system/mps_memory_allocated_mb", mps_allocated, step)

    def log_embedding_analysis(self, embeddings: torch.Tensor, step: int, prefix: str = "embeddings"):
        """Log embedding analysis statistics.

        Args:
            embeddings: Embedding tensor (vocab_size, embedding_dim)
            step: Global training step
            prefix: Prefix for metric names
        """
        with torch.no_grad():
            # Basic statistics
            self.writer.add_scalar(f"{prefix}/mean", embeddings.mean(), step)
            self.writer.add_scalar(f"{prefix}/std", embeddings.std(), step)
            self.writer.add_scalar(f"{prefix}/norm_mean", embeddings.norm(dim=1).mean(), step)
            self.writer.add_scalar(f"{prefix}/norm_std", embeddings.norm(dim=1).std(), step)

            # Cosine similarity statistics
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
            off_diagonal_similarities = similarity_matrix[mask]
            
            self.writer.add_scalar(f"{prefix}/cosine_sim_mean", off_diagonal_similarities.mean(), step)
            self.writer.add_scalar(f"{prefix}/cosine_sim_std", off_diagonal_similarities.std(), step)

    def log_model_info(self, model: nn.Module, sample_input: torch.Tensor):
        """Log model architecture information.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor for model graph
        """
        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.writer.add_text("model/total_parameters", str(total_params))
        self.writer.add_text("model/trainable_parameters", str(trainable_params))
        self.writer.add_text("model/architecture", str(model))

        # Model graph (if possible)
        try:
            self.writer.add_graph(model, sample_input)
        except Exception:
            pass  # Some models may not be traceable

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
