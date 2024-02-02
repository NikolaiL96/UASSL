import os
import time
import logging

import torch

from torch.utils.tensorboard import SummaryWriter

from utils import check_existing_model
from .utils import get_params_
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import entropy
import torch.nn as nn
from torchmetrics.functional.classification import binary_auroc as auc

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("Logger in SSL-trainer.")


class SupervisedTrainer(object):
    def __init__(self, model, ssl_data, data_root, device='cuda', save_root="", checkpoint_path=None, fine_tune="",
                 distribution=None, train_data='cifar10', clip=0., clip_type="Norm"):

        super().__init__()
        # Define device
        self.device = torch.device(device)
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip = clip
        self.clip_type = clip_type

        self.train_data = train_data
        self.data_root = data_root
        self.distribution = distribution
        self.ssl_data = ssl_data
        self.loss_fn = nn.CrossEntropyLoss()

        # Init
        self.epoch = 0

        self.loss_hist = []

        self._iter_scheduler = False
        self._hist_lr = []

        self.fine_tune = fine_tune
        self.save_root = save_root
        self.checkpoint_path = checkpoint_path

        self.environment = os.getenv('SLURM_JOB_PARTITION') if self.use_amp else "gpu-test"

        # Setup tensorboard logging of the training
        if self.environment != "gpu-test":
            self.tb_logger = SummaryWriter(log_dir=os.path.join(save_root, 'tb_logs'))

        # Model
        self.model = model.to(self.device)

        # Define data
        self.data = ssl_data

    def evaluate(self,):
        Recall, Auroc = [], []
        total_top1, total_num = 0.0, 0
        for x, labels in self.data.test_dl:
            x, labels = x.to(self.device), labels.to(self.device)
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    output = self.model(x)

                pred = torch.argmax(output, dim=-1)
                unc = entropy(nn.Softmax(dim=-1)(output).cpu().numpy(), axis=1)
                is_same_class = (pred == labels).float()
                auroc = auc(-torch.as_tensor(unc, device=self.device), is_same_class.int()).item()

                total_num += labels.size(0)
                total_top1 += (pred == labels).float().sum().item()

                Recall.append(is_same_class.mean())
                Auroc.append(auroc)

        Recall = torch.stack(Recall, 0)
        Auroc = torch.Tensor(Auroc)
        acc = total_top1 / total_num * 100
        return Recall.mean(), Auroc.mean(), acc

    def train_epoch(self, epoch_id):
        # make sure that things get set properly
        self.model.train()
        self.model.requires_grad_(True)

        nan_loss_counter = 0
        if epoch_id == 0:
            loading_time = 0.
            forward_time = 0.
            backward_time = 0.
            current_timestep = time.time()
        for i, (x, y) in enumerate(self.data.train_dl):
            x, y = x.to(self.device), y.to(self.device)
            if epoch_id == 0:
                loading_time += time.time() - current_timestep
                current_timestep = time.time()

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(x)
                loss = self.loss_fn(output, y)

            if epoch_id == 0:
                forward_time += time.time() - current_timestep
                current_timestep = time.time()

            # Save stats
            self._epoch_loss += loss.detach()

            # Backward pass
            self.optimizer.zero_grad()
            # GradScaling to avoid underflow when using autocast
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # save learning rate
            self._hist_lr.append(self.scheduler.get_last_lr())

            if self.scheduler and self._iter_scheduler:
                # Scheduler every iteration for cosine decay
                self.scheduler.step()

            # Save loss
            self._epoch_loss += loss

            if epoch_id == 0:
                backward_time += time.time() - current_timestep
                current_timestep = time.time()
        if epoch_id == 0:
            logger.info(f"Loading time {loading_time:.1f}s, Forward Time {forward_time:.1f}s, Backward Time "
                        f"{backward_time:.1f}s")

    def train(self, num_epochs, optimizer, optim_params, reduced_lr=False, **kwargs):

        # Check and Load existing model
        epoch_start, optim_state, sched_state = self.load_model(self.save_root, return_vals=True)

        # Extract training length
        self._train_len = len(self.data.train_dl)
        self._total_iters = num_epochs * self._train_len

        # Define Optimizer
        params = get_params_(self.fine_tune, self.model, reduced_lr, optim_params["lr"], logger)
        self.optimizer = optimizer(params, **optim_params)

        steps_per_epoch = 45000 // 256
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.1, steps_per_epoch=steps_per_epoch,
                                                             epochs=num_epochs)
        # Run Training
        for epoch in range(epoch_start, num_epochs):
            self.epoch = epoch
            self._epoch_loss = torch.zeros(1, device=self.device)


            start_time = time.time()

            self.train_epoch(epoch)

            # Log
            self.loss_hist.append(self._epoch_loss.item() / self._train_len)

            if self.environment != "gpu-test":
                self.tb_logger.add_scalar('loss/loss', self.loss_hist[-1], epoch)
                self.tb_logger.add_scalar('epoch_time', time.time() - start_time, epoch)

            logger.info(f'Epoch: {epoch}, Loss: {self.loss_hist[-1]:.3f}, Time epoch: {time.time() - start_time:0.1f}')

            if self.device.type == 'cuda':
                logger.debug(f'GPU Reserved {torch.cuda.memory_reserved(0) // 1e6}MB,'
                             f' Allocated {torch.cuda.memory_allocated(0) // 1e6}MB')

            if (epoch + 1) % 6 == 0:
                recall, auroc, acc= self.evaluate()

                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('kappa/linear_acc', acc, epoch)
                    self.tb_logger.add_scalar('kappa/AUROC_Dataset', auroc, epoch)
                    self.tb_logger.add_scalar('kappa/R@1_Dataset', recall, epoch)

                logger.info(f"Loss: {self.loss_hist[-1]:0.2f}, AUROC: {auroc:0.3f}, Recall: {recall:0.3f}, "
                            f"linear accuracy: {acc:0.1f}\n")

        # Save final model
        self.save_model(self.save_root, num_epochs)
        logger.info("Training completed. Final model saved.")

    def save_model(self, save_root, epoch):
        save_data = {'model': self.model.state_dict(),
                     'optim': self.optimizer.state_dict(),
                     'sched': self.scheduler.state_dict() if self.scheduler else None,
                     'loss_hist': self.loss_hist
                     }

        torch.save(save_data, os.path.join(save_root, f'epoch_{epoch:03}.tar'))

    def load_model(self, save_root, return_vals=False, ask_user=False):
        if self.checkpoint_path == None:
            return 0, None, None
        else:
            # Check for trained model
            epoch_start, saved_data = check_existing_model(save_root, self.device, ask_user=ask_user)
            self.epoch = epoch_start
            if saved_data is None and return_vals:
                return epoch_start, None, None
            else:
                self.model.load_state_dict(saved_data['model'])
                try:
                    self.loss_hist = saved_data['loss_hist']
                except Exception as e:
                    logger.error(f'Error loading model: {e}')

                if return_vals:
                    return epoch_start, saved_data['optim'], saved_data['sched']
