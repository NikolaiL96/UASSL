import os
import time
import logging

import torch
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from utils import check_existing_model
from .utils import get_params_
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import entropy, spearmanr
import torch.nn as nn
from torchmetrics.functional.classification import binary_auroc as auc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

from utils import load_dataset
from utils.utils import get_cifar10h

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

        # self.model.backbone_net.avgpool.register_forward_hook(self.get_activation('second_to_last'))

        # Define data
        self.data = ssl_data
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def evaluate(self,):
        Recall, Auroc, Auroc_norm = [], [], []
        total_top1, total_num = 0.0, 0
        uncertainty = np.empty(0)
        for x, labels in self.data.test_dl:
            x, labels = x.to(self.device), labels.to(self.device)
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    output = self.model(x)

                pred = torch.argmax(output, dim=-1)
                unc = entropy(nn.Softmax(dim=-1)(output).cpu().numpy(), axis=1)
                unc_norm = torch.linalg.norm(output, dim=1)
                is_same_class = (pred == labels).float()
                auroc = auc(-torch.as_tensor(unc, device=self.device), is_same_class.int()).item()
                auroc_norm = auc(-unc_norm, is_same_class.int()).item()

                total_num += labels.size(0)
                total_top1 += (pred == labels).float().sum().item()

                uncertainty = np.append(uncertainty, unc)

                Recall.append(is_same_class.mean())
                Auroc.append(auroc)
                Auroc_norm.append(auroc_norm)

        # entropy_uncertainty = entropy(uncertainty, axis=0)
        # print(f"Entropy uncertainty: {entropy_uncertainty}, Std uncertainty: {np.std(uncertainty)}")

        Recall = torch.stack(Recall, 0)
        Auroc = torch.Tensor(Auroc)
        Auroc_norm = torch.Tensor(Auroc_norm)
        acc = total_top1 / total_num * 100
        return Recall.mean(), Auroc.mean(), Auroc_norm.mean(), acc

    @torch.no_grad()
    def corrupted_img(self):
        self.model.eval()

        # Modified from https://github.com/mkirchhof/url/tree/main
        crop_min = torch.tensor(0.1, device=self.device)
        crop_max = torch.tensor(0.5, device=self.device)
        crop = torch.rand(len(self.ssl_data.test_dl.dataset), device=self.device) * (crop_max - crop_min) + crop_min

        unc, unc_c = np.empty(0), np.empty(0)

        for n, (x, target) in enumerate(self.ssl_data.test_dl):
            x, target = x.to(self.device), target.to(self.device)
            x_c = torch.zeros_like(x, device=self.device)
            c_sizes = torch.zeros(x.shape[0], device=self.device)

            for i in range(x.shape[0]):
                # Crop each image individually because torchvision cannot do it batch-wise
                crop_size = int(torch.round(min(x.shape[2], x.shape[3]) * crop[n * x.shape[0] + i]))
                c_sizes[i] = crop_size
                x_c[i] = TF.resize(TF.center_crop(x[i], [crop_size]), [x.shape[2], x.shape[3]], antialias=True)

            with autocast(enabled=self.use_amp):
                output = self.model(x)
                output_c = self.model(x_c)

            unc = np.append(unc, entropy(nn.Softmax(dim=-1)(output).cpu().numpy(), axis=1))
            unc_c = np.append(unc_c, entropy(nn.Softmax(dim=-1)(output_c).cpu().numpy(), axis=1))

        p_cropped = (unc > unc_c).mean()
        cor_cropped = spearmanr(-unc_c, crop.cpu().numpy())[0]

        print(f"p_c: {p_cropped}, corr_c {cor_cropped}")

        return p_cropped, cor_cropped

    @torch.no_grad()
    def get_cor(self):
        unc = np.empty(0)
        dl_kwargs = {"batch_size": 512, "shuffle": False, "num_workers": min(os.cpu_count(), 0)}
        data_cifar10h, *_ = load_dataset("cifar10", "./data/", augmentation_type="BYOL", dl_kwargs=dl_kwargs)

        for n, (x_test, _) in enumerate(data_cifar10h.test_dl):
            x_test = x_test.to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(x_test)

            unc = np.append(unc, entropy(nn.Softmax(dim=-1)(output).cpu().numpy(), axis=1))

        cifar10h = get_cifar10h()
        unc_h = entropy(cifar10h, axis=1)

        corr = np.corrcoef(unc, unc_h)
        rank_corr = spearmanr(unc, unc_h)[0]

        print(f"Corr_p: {corr[0, 1]}, Corr_H: {rank_corr}")
        return corr[0, 1], rank_corr


    def vis_t_SNE(self):

        self.model.eval()

        test_labels, test_uncertainty, test_loc = (), (), ()
        with torch.no_grad():
            for n, (x, labels) in enumerate(self.data.test_dl):
                x, labels = x.to(self.device), labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    output = self.model(x)

                uncertainty = entropy(nn.Softmax(dim=-1)(output).cpu().numpy(), axis=1)

                test_labels += (labels,)
                test_uncertainty += (torch.as_tensor(uncertainty),)
                test_loc += (output, )
                #test_loc += (self.activation["second_to_last"],)

                dataset = str(self.data.test_dl.dataset).split("\n")[0].split(" ")[1]

            feats, labels, unc = torch.cat(test_loc), torch.cat(test_labels), torch.cat(test_uncertainty)

            if dataset == "cifar10":
                idx = torch.randperm(unc.shape[0])[:4000]

            else:
                id = torch.tensor(np.random.choice(np.unique(labels.cpu().numpy()), 10), device=self.device)
                idx = torch.sum((labels[:, None] == id) * id[None], dim=-1).nonzero().squeeze()[:4000]
                idx = idx.cpu()


            feats, labels, unc = feats.squeeze().cpu(), labels.cpu(), unc.cpu()
            feats, labels, unc = feats[idx], labels[idx], unc[idx]

            unc_min = unc.min()
            unc_max = unc.max()
            # Exponential weighting of higher uncertainties for better visualization
            unc = torch.exp((unc - unc_min) / (unc_max - unc_min) * 3.5)

        # Perform t-SNE
        feats_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(feats)

        matplotlib.use('PDF')
        fig = plt.figure()
        plt.style.use('default')
        plt.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, s=unc, alpha=0.6)

        # Remove axis ticks and labels.
        plt.xticks([])
        plt.yticks([])

        id = os.getenv('SLURM_JOB_ID')
        name = f"t-SNE_{dataset}_Supervised"
        path = f"/home/lorenzni/imgs/{id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)

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
        for i, (x, y) in enumerate(self.data.train_eval_dl):
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

        self.scheduler = scheduler(self.optimizer, **scheduler_params)


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

            if (epoch + 1) % 10 == 0:
                recall, auroc, auroc_norm, acc = self.evaluate()

                if self.environment != "gpu-test":
                    self.tb_logger.add_scalar('kappa/linear_acc', acc, epoch)
                    self.tb_logger.add_scalar('kappa/AUROC_Dataset', auroc, epoch)
                    self.tb_logger.add_scalar('kappa/R@1_Dataset', recall, epoch)

                logger.info(f"Loss: {self.loss_hist[-1]:0.2f}, AUROC: {auroc:0.3f}, AUROC_Norm: {auroc_norm:0.3f}, Recall: {recall:0.3f}, "
                            f"linear accuracy: {acc:0.1f}\n")

        self.vis_t_SNE()
        self.corrupted_img()
        self.get_cor()

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
