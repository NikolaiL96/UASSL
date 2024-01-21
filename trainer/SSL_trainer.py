from os import path
import os
import time
from torch.utils.tensorboard import SummaryWriter

import torch
from torch.optim import lr_scheduler

from utils import check_existing_model, Validate
from utils import load_dataset

class SSL_Trainer(object):
    def __init__(self, model, ssl_data, data_root, device='cuda', save_root="", checkpoint_path=None, fine_tune="",
                 distribution=None, train_data='cifar10'):

        super().__init__()
        # Define device
        self.device = torch.device(device)

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
        self.distortion_loss_hist = []

        self._iter_scheduler = False
        self._hist_lr = []

        self.fine_tune = fine_tune
        self.save_root = save_root
        self.checkpoint_path = checkpoint_path

        self.tb_logger = SummaryWriter(log_dir=path.join(save_root, 'tb_logs'))

        # Model
        self.model = model.to(self.device)

        # Define data
        self.data = ssl_data

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
        for i, ((x1, x2), _) in enumerate(self.data.train_dl):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            if epoch_id == 0:
                loading_time += time.time() - current_timestep
                current_timestep = time.time()

            # Forward pass
            loss = self.model(x1, x2)

            if epoch_id == 0:
                forward_time += time.time() - current_timestep
                current_timestep = time.time()
            # Extract
            if isinstance(loss, tuple):
                ssl_loss, kl_loss, unc_loss = loss

            loss = ssl_loss + kl_loss + unc_loss

            # Save stats
            self._epoch_ssl_loss += ssl_loss.detach()
            self._epoch_kl_loss += kl_loss.detach()
            self._epoch_unc_loss += unc_loss.detach()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"We have a NAN or Inf Loss in either SSL {ssl_loss} or KL {kl_loss}")
                nan_loss_counter += 1
                # We just exclude this batch because of nan loss, but not to many times
                if nan_loss_counter < 10:
                    continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save learning rate
            self._hist_lr.append(self.scheduler.get_last_lr())

            if self.scheduler and self._iter_scheduler:
                # Scheduler every iteration for cosine deday
                self.scheduler.step()

            # Save loss
            self._epoch_loss += loss

            if epoch_id == 0:
                backward_time += time.time() - current_timestep
                current_timestep = time.time()
        if epoch_id == 0:
            print("For the first Epoch, we have the following Profiling results:")
            print(f"Loading time {loading_time:.1f}s, Forward Time {forward_time:.1f}s, Backward Time {backward_time:.1f}s")


    def train(self, num_epochs, optimizer, scheduler, optim_params, scheduler_params,
              warmup_epochs=10, iter_scheduler=True, evaluate_at=[100, 200, 400], verbose=True):

        # Check and Load existing model
        epoch_start, optim_state, sched_state = self.load_model(self.save_root, return_vals=True)

        # Extract training length
        self._train_len = len(self.data.train_dl)
        self._total_iters = num_epochs * self._train_len

        # Define Optimizer
        params = self.model.parameters()

        self.optimizer = optimizer(params, **optim_params)

        # Define Scheduler
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

        dl_kwargs = {"batch_size": 512, "shuffle": False, "num_workers": min(os.cpu_count(), 0)}
        data_val, _, in_chan = load_dataset(self.train_data, self.data_root, augmentation_type="BYOL", dl_kwargs=dl_kwargs)

        # Run Training
        for epoch in range(epoch_start, num_epochs):
            self.epoch = epoch
            self._epoch_loss = torch.zeros(1, device=self.device)
            self._epoch_ssl_loss = torch.zeros(1, device=self.device)
            self._epoch_unc_loss = torch.zeros(1, device=self.device)
            self._epoch_kl_loss = torch.zeros(1, device=self.device)

            start_time = time.time()

            self.train_epoch(epoch)

            if self.scheduler and not self._iter_scheduler:
                # Scheduler only every epoch
                self.scheduler.step()

            # Switch to new schedule after warmup period
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

            if self.tb_logger:
                self.tb_logger.add_scalar('loss/loss', self.loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/ssl_loss', self.ssl_loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/kl_loss', self.kl_loss_hist[-1], epoch)
                self.tb_logger.add_scalar('loss/unc_loss', self.unc_loss_hist[-1], epoch)

                self.tb_logger.add_scalar('epoch_time', time.time() - start_time, epoch)
                self.tb_logger.add_scalar('kappa/kappa_min', torch.min(self.model.kappa), epoch)
                self.tb_logger.add_scalar('kappa/kappa_max', torch.max(self.model.kappa), epoch)

                with torch.no_grad():
                    V = Validate(data=self.ssl_data, distribution=self.distribution, model=self.model, epoch=epoch,
                                 last_epoch=False, low_shot=False)

                    auroc, recall, knn, cor_corrupted, p_corrupted = V._get_metrics()
                    self.tb_logger.add_scalar('kappa/cor_corrupted', cor_corrupted, epoch)
                    self.tb_logger.add_scalar('kappa/p_corrupted', p_corrupted, epoch)

                    print(f"Auroc: {auroc.item():0.3f}, Recall: {recall.item():0.3f}, knn: {knn.item():0.1f}")

                    self.tb_logger.add_scalar('kappa/AUROC', auroc, epoch)
                    self.tb_logger.add_scalar('kappa/R@1', recall, epoch)
                    self.tb_logger.add_scalar('kappa/knn', knn, epoch)

            if verbose:
                print(f'Epoch: {epoch}, Loss: {self.loss_hist[-1]:0.4f}, AUROC: {auroc:0.3f}, Time epoch: {time.time() - start_time:0.1f}',
                      end='')
                if self.device.type == 'cuda':
                    print(f', GPU Reserved {torch.cuda.memory_reserved(0) // 1000000}MB,'
                          f' Allocated {torch.cuda.memory_allocated(0) // 1000000}MB', end='\n')

                print(f'SSL Loss: {self.ssl_loss_hist[-1]:0.4f}, Regularisation Loss: {self.kl_loss_hist[-1]:0.5f}, '
                      f'Uncertainty Loss: {self.unc_loss_hist[-1]:0.4f}')

            # Run evaluation
            if epoch == num_epochs - 1:
                V = Validate(data=self.ssl_data, distribution=self.distribution, model=self.model, epoch=epoch,
                             last_epoch=True, low_shot=False, plot_tsne=True)

                V_low_shot = Validate(data=self.ssl_data, distribution=self.distribution, model=self.model, epoch=epoch,
                                      last_epoch=False, plot_tsne=True, low_shot=True)

                _, _, _, cor_corrupted, p_corrupted = V._get_metrics()
                Auroc_100, Recall_100, knn_100, _, _ = V_low_shot._get_metrics()

                linear_acc_100 = V_low_shot._get_linear_probing()
                linear_acc_10 = V._get_linear_probing()

                cor_pearson, cor_spearman = V._get_cor()

                if not isinstance(cor_pearson, float):
                    cor_pearson = cor_pearson.item()

                if not isinstance(cor_spearman, float):
                    cor_spearman = cor_spearman.item()

                self.tb_logger.add_scalar('kappa/cor_corrupted', cor_corrupted, epoch)
                self.tb_logger.add_scalar('kappa/p_corrupted', p_corrupted, epoch)

                self.tb_logger.add_scalar('kappa/cor_pearson', cor_pearson, epoch)
                self.tb_logger.add_scalar('kappa/cor_spearman', cor_spearman, epoch)

                self.tb_logger.add_scalar('ZeroShot/AUROC_CIFAR100', Auroc_100, epoch)
                self.tb_logger.add_scalar('ZeroShot/R@1_CIFAR100', Recall_100, epoch)
                self.tb_logger.add_scalar('ZeroShot/knn_CIFAR100', knn_100, epoch)
                self.tb_logger.add_scalar('ZeroShot/Linear_accuracy_CIFAR100', linear_acc_100, epoch)
                self.tb_logger.add_scalar('ZeroShot/Linear_accuracy_CIFAR10', linear_acc_10, epoch)
                print(f"Low-Shot: AUROC={Auroc_100:0.4f}, R@1={Recall_100:0.3f}, KNN={knn_100.item():0.1f}")

            if (epoch) in evaluate_at:
                self.save_model(self.save_root, epoch + 1)

        # Save final model
        self.save_model(self.save_root, num_epochs)

        recall_cars, auroc_cars = V.get_zero_shot_metrics("cars196")
        self.tb_logger.add_scalar('ZeroShot/AUROC_CARS', auroc_cars, epoch)
        self.tb_logger.add_scalar('ZeroShot/R@1_CARS', recall_cars, epoch)

        recall_cup, auroc_cup = V.get_zero_shot_metrics("cup200")
        self.tb_logger.add_scalar('ZeroShot/AUROC_CUP', auroc_cup, epoch)
        self.tb_logger.add_scalar('ZeroShot/R@1_CUP', recall_cup, epoch)

        recall_sop, auroc_sop = V.get_zero_shot_metrics("sop")
        self.tb_logger.add_scalar('ZeroShot/AUROC_SOP', auroc_sop, epoch)
        self.tb_logger.add_scalar('ZeroShot/R@1_SOP', recall_sop, epoch)


    def save_model(self, save_root, epoch):
        save_data = {'model': self.model.state_dict(),
                     'optim': self.optimizer.state_dict(),
                     'sched': self.scheduler.state_dict() if self.scheduler else None,
                     'loss_hist': self.loss_hist,
                     'ssl_loss_hist': self.ssl_loss_hist,
                     'kl_loss_hist': self.kl_loss_hist,
                     'lr_hist': self._hist_lr}

        torch.save(save_data, path.join(save_root, f'epoch_{epoch:03}.tar'))

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
                    print(f'Error loading model: {e}')

                if return_vals:
                    return epoch_start, saved_data['optim'], saved_data['sched']
