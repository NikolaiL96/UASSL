from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import FashionMNIST

from torchvision.datasets import ImageFolder
from augmentation import (CURRICULUM_augmentations, BYOL_augmentaions, SimSiam_augmentaions, VICReg_augmentaions,
                          OneChannel_augmentaions, Supervised_augmentations)


CIFAR10_NORM = [[0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]]
CIFAR100_NORM = [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]]
# TODO Norm Taken from https://github.com/Skuldur/Oxford-IIIT-Pets-Pytorch/blob/master/Pytorch%20Image%20Classification.ipynb
OXFORD_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
FASHION_NORM = [[0.0], [1.0]]
CARS196_NORM = [[0.4699237327989344, 0.45947674570727726, 0.4544837074650634], [0.2633837294990095, 0.2622809594115579, 0.2670900292961565]]
SOP_NORM = [[0.5807119430341882, 0.5396427569883065, 0.5044110697472814], [0.2329339212837042, 0.23649191744643877, 0.237229328726736]]
CUP_NORM = [[0.4859091362359415, 0.4996146773549715, 0.43177199317237175], [0.17495981902740504, 0.17384660136682106, 0.18591599002787682]]

def get_norm(train_set):
    mean_r = 0
    mean_g = 0
    mean_b = 0
    std_r = std_g = std_b = 0
    k = 0
    for n, imgs in enumerate(train_set):
        imgs = imgs[0].squeeze().numpy()
        if k == 0:
            print(imgs.shape)

        # calculate mean over each channel (r,g,b)
        mean_r += imgs[0, :, :].mean()
        mean_g += imgs[1, :, :].mean()
        mean_b += imgs[2, :, :].mean()


        # calculate std over each channel (r,g,b)
        std_r += imgs[0, :, :].std()
        std_g += imgs[1, :, :].std()
        std_b += imgs[2, :, :].std()

        k += 1

    print(k)
    mean, std = (mean_r/k, mean_g/k, mean_b/k), (std_r/k, std_g/k, std_b/k)
    print(mean, std)
    return mean, std



def load_dataset(dataset, data_root, augmentation_type="BYOL", dl_kwargs=None):
    # Define datasets
    ad_ds = None
    if dataset == 'cifar10':
        ssl_data = SSL_CIFAR10(data_root, augmentation_type, CIFAR10_NORM, dl_kwargs)
        in_channels = 3
    elif dataset == 'cifar100':
        ssl_data = SSL_CIFAR100(data_root, augmentation_type, CIFAR100_NORM, dl_kwargs)
        in_channels = 3
    elif dataset == 'fashionmnist':
        ssl_data = SSL_FashionMNIST(data_root, augmentation_type, FASHION_NORM, dl_kwargs)
        in_channels = 1
    elif dataset == "cars196":
        ssl_data = SSL_CARS196(data_root, augmentation_type, CARS196_NORM, dl_kwargs)
        in_channels = 3
    elif dataset == "sop":
        ssl_data = SSL_SOP(data_root, augmentation_type, SOP_NORM, dl_kwargs)
        in_channels = 3
    elif dataset == "cup200":
        ssl_data = SSL_CUP(data_root, augmentation_type, CUP_NORM, dl_kwargs)
        in_channels = 3
    else:
        raise ValueError('Unknown dataset: ' + dataset)
    return ssl_data, ad_ds, in_channels


class SSL_CIFAR10(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):
        assert augmentation in [
            "BYOL",
            "SimSiam",
            "VICReg",
            "Curriculum",
            "Supervised",
        ], "augmentation must be in ['BYOL', 'SimSiam', 'VICReg','Curriculum', 'Supervised']"
        self.augmentation = augmentation
        # Define Augmentations
        if augmentation == "BYOL":
            train_transf = BYOL_augmentaions(image_size=32, normalize=normalisation)
        elif augmentation == "SimSiam":
            train_transf = SimSiam_augmentaions(image_size=32, normalize=normalisation)
        elif augmentation == "VICReg":
            train_transf = VICReg_augmentaions(image_size=32, normalize=normalisation)
        elif augmentation == "Curriculum":
            train_transf = CURRICULUM_augmentations(image_size=32, magnitude=0.0, normalize=normalisation)
        elif augmentation == "Supervised":
            train_transf = Supervised_augmentations(image_size=32, normalize=normalisation)

        train_eval_transf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalisation),
            ]
        )

        test_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalisation)]
        )

        norm_transf = transforms.Compose(
            [transforms.ToTensor()]
        )

        # Define Datasets
        train_ds = CIFAR10(root=data_root, train=True, download=True, transform=train_transf)
        train_eval_ds = CIFAR10(root=data_root, train=True, transform=train_eval_transf, download=True)
        train_plain_dl = CIFAR10(root=data_root, train=True, transform=test_transf, download=True)
        test_ds = CIFAR10(root=data_root, train=False, transform=test_transf, download=True)
        train_norm_ds = CIFAR10(root=data_root, train=True, download=True, transform=norm_transf)

        # Define Dataloaders
        self.train_dl = DataLoader(train_ds, drop_last=True, **dl_kwargs)
        self.train_eval_dl = DataLoader(train_eval_ds, drop_last=False, **dl_kwargs)
        self.train_plain_dl = DataLoader(train_plain_dl, drop_last=False, **dl_kwargs)
        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)
        self.train_norm_ds = DataLoader(train_norm_ds, drop_last=False, **dl_kwargs)
        self.normalisation = normalisation
    def normalization_params(self):
        return self.normalisation  # CIFAR10_NORM

    def update_Curriculum_transforms(self, magnitude=1.0):
        assert (
            self.augmentation == "Curriculum"
        ), "We should not update the transforms when we do not use curriculum transforms"
        self.train_dl.dataset.transform = CURRICULUM_augmentations(
            image_size=32, magnitude=magnitude, normalize=CIFAR10_NORM
        )

class SSL_CARS196(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):
        data_root += "cars196/"

        # Define Augmentations
        if augmentation == "BYOL":
            train_transf = BYOL_augmentaions(image_size=224, normalize=normalisation)
        elif augmentation == "SimSiam":
            train_transf = SimSiam_augmentaions(image_size=224, normalize=normalisation)
        elif augmentation == "VICReg":
            train_transf = VICReg_augmentaions(image_size=224, normalize=normalisation)
        elif augmentation == "Curriculum":
            train_transf = CURRICULUM_augmentations(
                image_size=224, magnitude=0.0, normalize=normalisation
            )

        train_eval_transf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalisation),
                transforms.Resize(size=(224, 224), antialias=True)
            ]
        )

        test_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalisation), transforms.Resize(size=(224, 224), antialias=True)]
        )

        norm_transf = transforms.Compose(
            [transforms.ToTensor() , transforms.Resize(size=(224, 224), antialias=True)]
        )

        # Define Datasets
        train_ds = ImageFolder(root=data_root + "train/", transform=train_transf
        )
        train_eval_ds = ImageFolder(root=data_root + 'train/', transform=train_eval_transf
        )
        train_plain_dl = ImageFolder(root=data_root + 'train/', transform=test_transf
        )
        test_ds = ImageFolder(root=data_root + 'test/', transform=test_transf
        )
        train_norm_ds = ImageFolder(root=data_root + 'train/', transform=norm_transf
        )

        # Define Dataloaders
        self.train_dl = DataLoader(train_ds, drop_last=True, **dl_kwargs)
        self.train_eval_dl = DataLoader(train_eval_ds, drop_last=False, **dl_kwargs)
        self.train_plain_dl = DataLoader(train_plain_dl, drop_last=False, **dl_kwargs)
        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)
        self.train_norm_ds = DataLoader(train_norm_ds, drop_last=False, **dl_kwargs)
        self.normalisation = normalisation

    def normalization_params(self):
        return self.normalisation

    def update_Curriculum_transforms(self, magnitude=1.0):
        assert (
            self.augmentation == "Curriculum"
        ), "We should not update the transforms when we do not use curriculum transforms"
        self.train_dl.dataset.transform = CURRICULUM_augmentations(
            image_size=32, magnitude=magnitude, normalize=self.normalisation
        )

class SSL_CUP(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):

        test_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalisation), transforms.Resize(size=(224, 224), antialias=True)]
        )

        norm_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(224, 224), antialias=True)]
        )


        # Define Datasets

        test_ds = ImageFolder(root=data_root, transform=test_transf)
        train_norm_ds = ImageFolder(root=data_root, transform=norm_transf)

        # Define Dataloaders

        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)
        self.train_norm_ds = DataLoader(train_norm_ds, drop_last=False, **dl_kwargs)
        self.normalisation = normalisation

    def normalization_params(self):
        return self.normalisation

    def update_Curriculum_transforms(self, magnitude=1.0):
        assert (
            self.augmentation == "Curriculum"
        ), "We should not update the transforms when we do not use curriculum transforms"
        self.train_dl.dataset.transform = CURRICULUM_augmentations(
            image_size=64, magnitude=magnitude, normalize=self.normalisation
        )

class SSL_SOP(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):

        test_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalisation), transforms.Resize(size=(224, 224), antialias=True)]
        )

        norm_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(224, 224))]
        )

        # Define Datasets
        test_ds = ImageFolder(root=data_root, transform=test_transf)
        train_norm_ds = ImageFolder(root=data_root, transform=norm_transf)

        # Define Dataloaders
        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)
        self.train_norm_ds = DataLoader(train_norm_ds, drop_last=False, **dl_kwargs)
        self.normalisation = normalisation

    def normalization_params(self):
        return self.normalisation

    def update_Curriculum_transforms(self, magnitude=1.0):
        assert (
            self.augmentation == "Curriculum"
        ), "We should not update the transforms when we do not use curriculum transforms"
        self.train_dl.dataset.transform = CURRICULUM_augmentations(
            image_size=64, magnitude=magnitude, normalize=self.normalisation
        )

class SSL_FashionMNIST(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):
        self.augmentation = augmentation
        train_transf = OneChannel_augmentaions(image_size=28)

        train_eval_transf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        test_transf = transforms.Compose([transforms.ToTensor()])

        # Define Datasets
        train_ds = FashionMNIST(
            root=data_root, train=True, download=True, transform=train_transf
        )
        train_eval_ds = FashionMNIST(
            root=data_root, train=True, transform=train_eval_transf, download=True
        )
        test_ds = FashionMNIST(
            root=data_root, train=False, transform=test_transf, download=True
        )

        # Define Dataloaders
        self.train_dl = DataLoader(train_ds, drop_last=True, **dl_kwargs)
        self.train_eval_dl = DataLoader(train_eval_ds, drop_last=False, **dl_kwargs)
        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)
        self.normalisation = normalisation
    def normalization_params(self):
        return self.normalisation  # FASHION_NORM


class SSL_CIFAR100(object):
    def __init__(self, data_root, augmentation, normalisation, dl_kwargs):
        assert augmentation in [
            "BYOL",
            "SimSiam",
            "VICReg",
            "Supervised"
        ], "augmentation must be in ['BYOL', 'SimSiam', 'VICReg', 'Supervised']"
        self.augmentation = augmentation
        self.normalisation = normalisation
        # Define Augmentations
        if augmentation == "BYOL":
            train_transf = BYOL_augmentaions(image_size=32, normalize=self.normalisation)
        elif augmentation == "SimSiam":
            train_transf = SimSiam_augmentaions(image_size=32, normalize=self.normalisation)
        elif augmentation == "VICReg":
            train_transf = VICReg_augmentaions(image_size=32, normalize=self.normalisation)
        elif augmentation == "Supervised":
            train_transf = Supervised_augmentations(image_size=32, normalize=normalisation)

        train_eval_transf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*self.normalisation),
            ]
        )

        test_transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*self.normalisation)]
        )

        # Define Datasets
        train_ds = CIFAR100(
            root=data_root, train=True, download=True, transform=train_transf
        )
        train_eval_ds = CIFAR100(
            root=data_root, train=True, transform=train_eval_transf, download=True
        )
        test_ds = CIFAR100(
            root=data_root, train=False, transform=test_transf, download=True
        )

        # Define Dataloaders
        self.train_dl = DataLoader(train_ds, drop_last=True, **dl_kwargs)
        self.train_eval_dl = DataLoader(train_eval_ds, drop_last=False, **dl_kwargs)
        self.test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)

    def normalization_params(self):
        return self.normalisation

# if __name__ == "__main__":
#
#     # ssl_data = SSL_CARS196(data_root="/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Code/data", augmentation="BYOL", normalisation=CIFAR10_NORM,
#     #                        dl_kwargs={"batch_size": 1, "shuffle": False, "num_workers": min(os.cpu_count(), 0)})
#     #
#     # get_norm(ssl_data.train_norm_ds)
#     ssl_data_sop = SSL_SOP(augmentation="BYOL", normalisation=SOP_NORM, dl_kwargs={"batch_size": 1, "shuffle": False,
#                                                                    "num_workers": min(os.cpu_count(), 0)})
#
#     ssl_data_cup = SSL_CUP(augmentation="BYOL", normalisation=CUP_NORM, dl_kwargs={"batch_size": 1, "shuffle": False,
#                                                                    "num_workers": min(os.cpu_count(), 0)})
#
#     m_sop, std_sop = get_norm(ssl_data_sop.train_norm_ds)
#     print(f"SOP: {m_sop}, {std_sop}")
#
#     m_cup, std_cup = get_norm(ssl_data_cup.train_norm_ds)
#     print(f"SOP: {m_cup}, {std_cup}")
