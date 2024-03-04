
import logging
import torch

from utils import check_existing_model, Validate, Linear_Protocoler, plot_kappa_class


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("Logger in SSL-trainer.")


class Evaluator(object):
    def __init__(self, model, ssl_data, eval_params, device='cuda', distribution=None, epoch=1000):

        super().__init__()
        # Define device
        self.distribution = distribution
        self.device = torch.device(device)
        # Model
        self.model = model.to(self.device)

        # Define data
        self.data = ssl_data

        self.model.eval()
        self.model.requires_grad_(False)

        plot_kappa_class(self.model, self.device, ssl_data)

        validate = Validate(data=self.data, device=self.device, distribution=self.distribution,
                            model=self.model, epoch=epoch, last_epoch=True, low_shot=False, plot_tsne=True)


        cor_corrupted, p_corrupted = validate.get_metrics()
        _, cor_spearman = validate.get_cor()

        print(f"Corr_C: {cor_corrupted}, Corr_H: {cor_spearman}, P_C: {p_corrupted}")

        linear_acc_10 = validate.get_linear_probing(eval_params=eval_params)

        # Run evaluation
        validate_low_shot = Validate(data=self.data, device=self.device, distribution=self.distribution,
                                     model=self.model, epoch=epoch, last_epoch=False, plot_tsne=False,
                                     low_shot=True)

        _, _, knn, recall, auroc = self.evaluate(eval_params=eval_params)

        print(f"Linear Accuracy: {linear_acc_10}, KNN: {knn}, Recall: {recall}, AUROC: {auroc}")

        recall_cifar100, auroc_cifar100 = validate_low_shot.recall_auroc()
        linear_acc_100 = validate_low_shot.get_linear_probing(eval_params=eval_params)
        print(f"Linear Accuracy C100: {linear_acc_100}, Recall C100: {recall_cifar100}, AUROC C100: {auroc_cifar100}")

    def evaluate(self, eval_params):
        # Linear protocol
        evaluator = Linear_Protocoler(self.model.backbone_net, repre_dim=self.model.rep_dim, device=self.device,
                                      eval_params=eval_params, distribution=self.distribution)
        # knn accuracy
        knn = evaluator.knn_accuracy(self.data.train_eval_dl, self.data.test_dl)

        # R@1 and R-AUROC
        recall, auc, R2, A2 = evaluator.recall_auroc(self.data.test_dl, get_entropy=True)
        return recall, auc, knn, R2, A2


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
