import os
import time
import logging

import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils import check_existing_model, Validate, Linear_Protocoler
from .utils import grad_clip_hook_, get_params_
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("Logger in SSL-trainer.")


class SSL_Trainer(object):
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

        # Init
        self.epoch = 0
        self.loss_hist = []
        self.ssl_loss_hist = []
        self.kl_loss_hist = []
        self.unc_loss_hist = []
        self.dist_std_hist_stats = {'min': [], 'max': [], 'mean': [], 'diversity': []}

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

    def evaluate(self, eval_params):
        # Linear protocol
        evaluator = Linear_Protocoler(self.model.backbone_net, repre_dim=self.model.rep_dim, device=self.device,
                                      eval_params=eval_params, distribution=self.distribution)
        # knn accuracy
        knn = evaluator.knn_accuracy(self.data.train_eval_dl, self.data.test_dl)

        # R@1 and R-AUROC
        recall, auc, R2, A2 = evaluator.recall_auroc(self.data.test_dl)
        return recall, auc, knn, R2, A2

    def train_epoch(self, epoch_id):
        # make sure that things get set properly
        self.model.train()
        self.model.requires_grad_(True)

        # TODO think if can be deleted
        if "Hook" in self.clip_type:
            grad_clip_hook_(self.model.backbone_net, clip=self.clip, clip_type=self.clip_type)

        nan_loss_counter = 0
        if epoch_id == 0:
            loading_time = 0.
            forward_time = 0.
            backward_time = 0.
            current_timestep = time.time()
        for i, ((x1, x2), _) in enumerate(self.data.train_dl):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            if epoch_id == 0:
                loading_time += time.time() - current_timestep
                current_timestep = time.time()

            # Forward pass
            with autocast(enabled=self.use_amp):
                loss = self.model(x1, x2, epoch_id)

            if epoch_id == 0:
                forward_time += time.time() - current_timestep
                current_timestep = time.time()

            # Extract
            ssl_loss, kl_loss, unc_loss, (dist1, dist2) = loss
            loss = ssl_loss + kl_loss + unc_loss

            # Save stats
            self._epoch_ssl_loss += ssl_loss.detach()
            self._epoch_kl_loss += kl_loss.detach()
            self._epoch_unc_loss += unc_loss.detach()
            self._dist_std_stats['min'] = dist1.scale.min().item()
            self._dist_std_stats['max'] = dist1.scale.max().item()
            self._dist_std_stats['mean'] += dist1.scale.mean().item()
            self._dist_std_stats['diversity'] += dist1.scale.std().item()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"We have a NAN or Inf Loss in either SSL {ssl_loss} or KL {kl_loss}")
                nan_loss_counter += 1
                # We just exclude this batch because of nan loss, but not to many times
                if nan_loss_counter < 10:
                    continue
                else:
                    raise ValueError("More then ten Nan detected in SSL or KL-loss!")

            # Backward pass
            self.optimizer.zero_grad()
            # GradScaling to avoid underflow when using autocast
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Prevent exploding kappa by clipping gradients
            if self.clip_type == "Norm":
                torch.nn.utils.clip_grad_norm_(self.model.backbone_net.parameters(), self.clip)
                if self.clip == 0.:
                    logger.warning(f"clip_type Norm but clip is 0.")

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

    def train(self, num_epochs, optimizer, scheduler, optim_params, scheduler_params, eval_params, evaluate_at,
              warmup_epochs=10, iter_scheduler=True, reduced_lr=False):

        # Check and Load existing model
        epoch_start, optim_state, sched_state = self.load_model(self.save_root, return_vals=True)

        # Extract training length
        self._train_len = len(self.data.train_dl)
        self._total_iters = num_epochs * self._train_len

        # Define Optimizer
        params = get_params_(self.fine_tune, self.model, reduced_lr, optim_params["lr"], logger)
        self.optimizer = optimizer(params, **optim_params)

        # Define Scheduler
        # TODO cleanup _ider_schedular
        if warmup_epochs and epoch_start < warmup_epochs:
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer,
                                                   lambda it: (it + 1) / (warmup_epochs * self._train_len))
            self._iter_scheduler = True
        else:
            if scheduler:
                self.scheduler = scheduler(self.optimizer, **scheduler_params)
                self._iter_scheduler = iter_scheduler
            else:
                self.scheduler = scheduler

        # Run Training
        for epoch in range(epoch_start, num_epochs):
            self.epoch = epoch
            self._epoch_loss = torch.zeros(1, device=self.device)
            self._epoch_ssl_loss = torch.zeros(1, device=self.device)
            self._epoch_unc_loss = torch.zeros(1, device=self.device)
            self._epoch_kl_loss = torch.zeros(1, device=self.device)
            self._dist_std_stats = {'min': 1e8, 'max': 1e8, 'mean': torch.zeros(1, device=self.device),
                                    'diversity': torch.zeros(1, device=self.device)}

            start_time = time.time()

            self.train_epoch(epoch)

            # TODO think if can be deleted
            if self.scheduler and not self._iter_scheduler:
                # Scheduler only every epoch
                self.scheduler.step()

            # Switch to new schedule after warmup period
            # TODO clean up if statement
            if warmup_epochs and epoch + 1 == warmup_epochs:
                if scheduler:
                    self.scheduler = scheduler(self.optimizer, **scheduler_params)
                    self._iter_scheduler = iter_scheduler
                else:  # scheduler = None
                    self.scheduler = scheduler

            # Log
            self.loss_hist.append(self._epoch_loss.item() / self._train_len)
            self.ssl_loss_hist.append(self._epoch_ssl_loss.item() / self._train_len)
            self.kl_loss_hist.append(self._epoch_kl_loss.item() / self._train_len)
            self.unc_loss_hist.append(self._epoch_unc_loss.item() / self._train_len)

            # TODO function for logging and saving results
            self.dist_std_hist_stats['min'].append(self._dist_std_stats['min'])
            self.dist_std_hist_stats['max'].append(self._dist_std_stats['max'])
            self.dist_std_hist_stats['mean'].append(self._dist_std_stats['mean'].item() / self._train_len)
            self.dist_std_hist_stats['diversity'].append(self._dist_std_stats['diversity'] / self._train_len)

            if self.environment != "gpu-test":
                self.tb_logger.add_scalar('loss/loss', self.loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/ssl_loss', self.ssl_loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/kl_loss', self.kl_loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/unc_loss', self.unc_loss_hist[-1], epoch)

                self.tb_logger.add_scalar('epoch_time', time.time() - start_time, epoch)

                self.tb_logger.add_scalar('kappa/kappa_min', self.dist_std_hist_stats["min"][-1], epoch)
                self.tb_logger.add_scalar('kappa/kappa_max', self.dist_std_hist_stats["max"][-1], epoch)

            # TODO think if cuda synchronisation should be used
            logger.info(f'Epoch: {epoch}, Time epoch: {time.time() - start_time:0.1f}')

            if self.device.type == 'cuda':
                logger.debug(f'GPU Reserved {torch.cuda.memory_reserved(0) // 1e6}MB,'
                             f' Allocated {torch.cuda.memory_allocated(0) // 1e6}MB')

            logger.info(
                f'SSL Loss: {self.ssl_loss_hist[-1]:0.4f}, Regularisation Loss: {self.kl_loss_hist[-1]:0.5f}, '
                f'Uncertainty Loss: {self.unc_loss_hist[-1]:0.4f}, Std [mean/min/max]: '
                f'[{self.dist_std_hist_stats["mean"][-1]:0.2f}, {self.dist_std_hist_stats["min"][-1]:0.2f}, '
                f'{self.dist_std_hist_stats["max"][-1]:0.2f}]')

            # TODO validation time
            if (epoch + 1) % 6 == 0:
                R2, A2, knn, recall, auroc = self.evaluate(eval_params=eval_params)

                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('kappa/AUROC', A2, epoch)
                    self.tb_logger.add_scalar('kappa/R@1', R2, epoch)
                    self.tb_logger.add_scalar('kappa/knn', knn, epoch)
                    self.tb_logger.add_scalar('kappa/AUROC_Dataset', auroc, epoch)
                    self.tb_logger.add_scalar('kappa/R@1_Dataset', recall, epoch)

                logger.info(f"Loss: {self.loss_hist[-1]:0.2f}, AUROC: {auroc:0.3f}, Recall: {recall:0.3f}, "
                            f"knn: {knn:0.1f}\n")

            if (epoch + 1) in evaluate_at:
                self.save_model(self.save_root, epoch + 1)

                validate = Validate(data=self.ssl_data, device=self.device, distribution=self.distribution,
                                    model=self.model, epoch=epoch, last_epoch=False, low_shot=False, plot_tsne=False)

                cor_corrupted, p_corrupted = validate.get_metrics()
                cor_pearson, cor_spearman = validate.get_cor()
                linear_acc_10 = validate.get_linear_probing(eval_params=eval_params)


                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('kappa/cor_corrupted', cor_corrupted, epoch)
                    self.tb_logger.add_scalar('kappa/p_corrupted', p_corrupted, epoch)

                    self.tb_logger.add_scalar('kappa/cor_pearson', cor_pearson, epoch)
                    self.tb_logger.add_scalar('kappa/cor_spearman', cor_spearman, epoch)
                    self.tb_logger.add_scalar('ZeroShot/Linear_accuracy_CIFAR10', linear_acc_10, epoch)

            # Run evaluation
            if epoch == num_epochs - 1:
                validate = Validate(data=self.ssl_data, device=self.device, distribution=self.distribution,
                                    model=self.model, epoch=epoch, last_epoch=True, low_shot=False, plot_tsne=True)

                validate_low_shot = Validate(data=self.ssl_data, device=self.device, distribution=self.distribution,
                                             model=self.model, epoch=epoch, last_epoch=False, plot_tsne=True,
                                             low_shot=True)

                cor_corrupted, p_corrupted = validate.get_metrics()
                cor_pearson, cor_spearman = validate.get_cor()
                linear_acc_10 = validate.get_linear_probing(eval_params=eval_params)

                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('kappa/cor_corrupted', cor_corrupted, epoch)
                    self.tb_logger.add_scalar('kappa/p_corrupted', p_corrupted, epoch)

                    self.tb_logger.add_scalar('kappa/cor_pearson', cor_pearson, epoch)
                    self.tb_logger.add_scalar('kappa/cor_spearman', cor_spearman, epoch)
                    self.tb_logger.add_scalar('ZeroShot/Linear_accuracy_CIFAR10', linear_acc_10, epoch)

                recall_cifar100, auroc_cifar100 = validate_low_shot.recall_auroc()
                linear_acc_100 = validate_low_shot.get_linear_probing(eval_params=eval_params)

                print(f"AUROC100: {auroc_cifar100:0.3f}, Recall100: {recall_cifar100:0.3f}, LinAcc100: {linear_acc_100:0.1f}\n")

                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('ZeroShot/AUROC_CIFAR100', auroc_cifar100, epoch)
                    self.tb_logger.add_scalar('ZeroShot/R@1_CIFAR100', recall_cifar100, epoch)
                    self.tb_logger.add_scalar('ZeroShot/Linear_accuracy_CIFAR100', linear_acc_100, epoch)

        # Save final model
        self.save_model(self.save_root, num_epochs)
        logger.info("Training completed. Final model saved.")

    def save_model(self, save_root, epoch):
        save_data = {'model': self.model.state_dict(),
                     'optim': self.optimizer.state_dict(),
                     'sched': self.scheduler.state_dict() if self.scheduler else None,
                     'loss_hist': self.loss_hist,
                     'ssl_loss_hist': self.ssl_loss_hist,
                     'kl_loss_hist': self.kl_loss_hist,
                     'lr_hist': self._hist_lr}

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
                    self.ssl_loss_hist = saved_data['ssl_loss_hist']
                    self.kl_loss_hist = saved_data['kl_loss_hist']
                    self._hist_lr = saved_data['lr_hist']
                except Exception as e:
                    logger.error(f'Error loading model: {e}')

                if return_vals:
                    return epoch_start, saved_data['optim'], saved_data['sched']
