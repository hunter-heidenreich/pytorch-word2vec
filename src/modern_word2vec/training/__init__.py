"""Training utilities and trainer class."""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modern_word2vec.config import TrainConfig


class Trainer:
    """Trainer class for Word2Vec models."""

    def __init__(self, model: nn.Module, device: torch.device, config: TrainConfig):
        """Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.device = device
        self.config = config

        # Only create criterion for full softmax models
        # Hierarchical softmax models compute loss internally
        if (
            hasattr(model, "output_layer_type")
            and model.output_layer_type == "hierarchical_softmax"
        ):
            self.criterion = None
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = self._build_optimizer()

        # Set up learning rate scheduler for linear decay (Word2Vec paper requirement)
        self.scheduler = None

        # Initialize TensorBoard logger if enabled
        self.tb_logger = None
        if config.tensorboard:
            try:
                from modern_word2vec.utils.tensorboard_logger import TensorBoardLogger
                
                # Create experiment name from config
                experiment_name = f"{model.__class__.__name__}_{config.optimizer}_lr{config.learning_rate}_bs{config.batch_size}"
                
                self.tb_logger = TensorBoardLogger(
                    log_dir=config.tensorboard_dir,
                    log_gradients=config.log_gradients,
                    log_weights=config.log_weights,
                    log_system_stats=config.log_system_stats,
                    experiment_name=experiment_name,
                )
                
                # Log hyperparameters
                hparams = {
                    "model_type": model.__class__.__name__,
                    "embedding_dim": config.embedding_dim,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "optimizer": config.optimizer,
                    "epochs": config.epochs,
                    "weight_decay": config.weight_decay,
                    "grad_clip": config.grad_clip,
                    "mixed_precision": config.mixed_precision,
                }
                
                # We'll log final metrics at the end of training
                self.tb_logger.log_hyperparameters(hparams, {"final_loss": 0.0})
                
            except ImportError:
                print("Warning: TensorBoard dependencies not available. Skipping TensorBoard logging.")
                self.tb_logger = None

        # Apply model compilation if requested and available
        if config.compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass  # Silently fall back if compilation fails

        self.model.to(device)

        # Set up mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config.mixed_precision and device.type == "cuda"
        )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer based on configuration.

        Returns:
            Configured optimizer instance
        """
        optimizer_params = {
            "params": self.model.parameters(),
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }

        if self.config.optimizer.lower() == "sgd":
            return optim.SGD(**optimizer_params)
        return optim.Adam(**optimizer_params)

    def _setup_scheduler(self, total_steps: int) -> None:
        """Set up linear learning rate scheduler as per Word2Vec paper.

        The paper states: "We chose starting learning rate 0.025 and decreased it
        linearly, so that it approaches zero at the end of the last training epoch"

        Note: This is only applied when using SGD optimizer, as per the original paper.
        Adam and other adaptive optimizers manage their own learning rate dynamics.

        Args:
            total_steps: Total number of training steps across all epochs
        """
        # Only apply linear LR decay for SGD (as per original Word2Vec paper)
        if self.config.optimizer.lower() == "sgd":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,  # Start at full learning rate
                end_factor=0.0,  # End at zero learning rate
                total_iters=total_steps,
            )
        else:
            # For Adam and other adaptive optimizers, don't use LR scheduling
            self.scheduler = None

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Execute a single training step.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Loss value for this step
        """
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler.is_enabled():
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(inputs, targets)

            self.scaler.scale(loss).backward()

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(inputs, targets)
            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

        return loss.item()

    def _compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss using the appropriate method based on model type.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Loss tensor
        """
        # Check if model uses hierarchical softmax
        if (
            hasattr(self.model, "output_layer_type")
            and self.model.output_layer_type == "hierarchical_softmax"
        ):
            # For hierarchical softmax, we need to compute input embeddings differently based on model type
            if (
                hasattr(self.model, "__class__")
                and "CBOW" in self.model.__class__.__name__
            ):
                # For CBOW: inputs are context words, we need to compute the averaged context vector
                # Handle padded inputs by masking out padding (zeros)
                in_embeds = self.model.in_embeddings(
                    inputs
                )  # (batch, max_context_len, embedding_dim)

                # Create mask for non-zero (non-padding) tokens
                mask = (
                    (inputs != 0).float().unsqueeze(-1)
                )  # (batch, max_context_len, 1)

                # Apply mask and compute mean only over non-padding tokens
                masked_embeds = (
                    in_embeds * mask
                )  # (batch, max_context_len, embedding_dim)
                sum_embeds = masked_embeds.sum(dim=1)  # (batch, embedding_dim)
                count_embeds = mask.sum(dim=1).clamp(
                    min=1
                )  # (batch, 1) - prevent division by zero
                input_embeddings = sum_embeds / count_embeds  # (batch, embedding_dim)
            else:
                # For Skip-gram: inputs are single center words
                input_embeddings = self.model.in_embeddings(
                    inputs
                )  # (batch, embedding_dim)

            return self.model.compute_loss(input_embeddings, targets)
        else:
            # Standard full softmax - model returns scores, we apply cross-entropy
            scores = self.model(inputs)
            return self.criterion(scores, targets)

    def train(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train the model.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with training statistics
        """
        # Calculate total steps for linear LR decay (Word2Vec paper requirement)
        total_steps = len(dataloader) * self.config.epochs
        self._setup_scheduler(total_steps)

        # Log model info to TensorBoard if enabled
        if self.tb_logger is not None:
            try:
                # Get a sample batch for model graph logging
                sample_batch = next(iter(dataloader))
                sample_input = sample_batch[0][:1].to(self.device)  # Single sample
                self.tb_logger.log_model_info(self.model, sample_input)
            except Exception:
                pass  # Skip if model graph logging fails

        steps = 0
        samples = 0
        running_loss = 0.0
        start_time = time.perf_counter()

        for epoch in range(self.config.epochs):
            self.model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            for inputs, targets in pbar:
                loss = self._train_step(inputs, targets)

                batch_size = inputs.shape[0]
                samples += int(batch_size)
                running_loss += loss
                steps += 1

                # Step the learning rate scheduler after each batch (linear decay)
                if self.scheduler is not None:
                    self.scheduler.step()

                # TensorBoard logging
                if self.tb_logger is not None and steps % self.config.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    
                    # Log basic training metrics
                    self.tb_logger.log_training_metrics(
                        step=steps,
                        loss=loss,
                        learning_rate=current_lr,
                        batch_size=batch_size,
                        epoch=epoch,
                    )
                    
                    # Log gradient statistics
                    self.tb_logger.log_gradient_stats(self.model, steps)
                    
                    # Log weight statistics
                    self.tb_logger.log_weight_stats(self.model, steps)
                    
                    # Log system statistics
                    self.tb_logger.log_system_stats(steps, self.device)
                    
                    # # Log embedding analysis (less frequently to save compute)
                    # if steps % (self.config.log_interval * 10) == 0:
                    #     try:
                    #         embeddings = self.model.in_embeddings.weight.detach()
                    #         self.tb_logger.log_embedding_analysis(embeddings, steps)
                    #     except Exception:
                    #         pass  # Skip if embedding analysis fails

                # Update progress bar with current loss and learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    {"loss": f"{running_loss / steps:.4f}", "lr": f"{current_lr:.6f}"}
                )

        total_time = time.perf_counter() - start_time

        # Final metrics
        final_metrics = {
            "avg_loss": running_loss / max(1, steps),
            "steps": steps,
            "samples": samples,
            "time_sec": total_time,
            "steps_per_sec": steps / max(total_time, 1e-9),
            "samples_per_sec": samples / max(total_time, 1e-9),
            "final_lr": self.optimizer.param_groups[0]["lr"],
        }

        # Log final metrics to TensorBoard
        if self.tb_logger is not None:
            for key, value in final_metrics.items():
                self.tb_logger.writer.add_scalar(f"final/{key}", value, steps)
            
            # Update hyperparameters with final loss
            try:
                hparams = {
                    "model_type": self.model.__class__.__name__,
                    "embedding_dim": self.config.embedding_dim,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "optimizer": self.config.optimizer,
                    "epochs": self.config.epochs,
                    "weight_decay": self.config.weight_decay,
                    "grad_clip": self.config.grad_clip,
                    "mixed_precision": self.config.mixed_precision,
                }
                self.tb_logger.log_hyperparameters(hparams, {"final_loss": final_metrics["avg_loss"]})
            except Exception:
                pass  # Skip if hyperparameter logging fails
            
            # Close TensorBoard logger
            self.tb_logger.close()

        return final_metrics
