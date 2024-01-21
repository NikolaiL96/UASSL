import os
import pathlib
import random
import shutil
from typing import Any, Callable, Optional, Tuple
import copy
from PIL import Image
import numpy as np
import pandas as pd
import torch

from torchvision.datasets import CIFAR10, VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import torchvision.transforms as TF


class FewShotCIFAR10(CIFAR10):
    def __init__(
            self, samples_per_class: int, selected_classes=None, seed: int = 0, **kwargs
    ):
        super().__init__(**kwargs)
        random.seed(seed)

        if selected_classes is None:
            selected_classes = list(range(10))

        sample_indices = []
        _targets = np.array(self.targets)

        for cix in selected_classes:
            _indices = np.argwhere(_targets == cix).reshape(-1).tolist()
            _indices = random.choices(_indices, k=samples_per_class)
            sample_indices.extend(_indices)

        # use to map the original class indices to the new one
        self._label_mapping = dict(zip(selected_classes, range(len(selected_classes))))
        # keep the original class indices for sanity check
        self._original_targets = _targets[sample_indices].tolist()

        # these are three important attributes of https://pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html
        self.data = self.data[sample_indices, :, :, :]
        self.targets = list(
            map(lambda i: self._label_mapping[i], self._original_targets)
        )
        self.classes = list(map(lambda i: self.classes[i], selected_classes))


class NoisyBg_OxfordPet(VisionDataset):
    """`Cutout Oxford-IIIT Pet Dataset with Specific Background  <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        small_test (bool, optional): Make the testset smaller and the trainset bigger. Default is True.
        target_levels (string, optional): Types of target level to use. Can be ``"breed"``
            or ``"species"`` (default), where spcies becomes a binary cat-vs-dog classification and breed is
            a fain-grained classification between 37 different breeds.
        img_size (int, optional): Size of the final image. Default is 128.
        pet_size (int, optional): Size of the pet image within the final image. Default is 48.
        background (string, optional): Which background zu use. Can be ``"white"`` (default), ``"noise"`` or
            ``"tar_corr_noise"``. In future: list of background images, correlated background colour and others.
        hard_cut (bool, optinal): Whether to cut out the randering pixels of pets as well. Default is False.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """
    # TODO: Add the possibility to upload costum list of backgrounds.
    # We might need to change the way we currently create the images: Expand the mask or something...

    _RESOURCES = (
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    )
    _VALID_TARGET_LEVEL = ("breed", "species")
    _VALID_BACKGROUND = ("big_noise", "noise", "tar_corr_noise", "white")
    _BROKEN_SEGMENTATION = [
        "Egyptian_Mau_20",
        "Egyptian_Mau_162",
        "Egyptian_Mau_165",
        "Egyptian_Mau_196",
        "Persian_259",
        "japanese_chin_199",
        "keeshond_7",
        "leonberger_18",
        "miniature_pinscher_14",
        "saint_bernard_15",
        "saint_bernard_60",
        "saint_bernard_78",
        "saint_bernard_108",
        "wheaten_terrier_195",
    ]

    def __init__(
            self,
            root: str,
            split: str = "trainval",
            small_test: bool = True,
            target_level: str = "species",
            img_size: int = 128,
            pet_size: int = 48,
            background: str = "white",
            hard_cut: bool = False,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            preload: bool = True,
    ) -> None:

        self._split = verify_str_arg(split.lower(), "split", ("trainval", "test"))
        self._small_test = small_test
        self._target_level = verify_str_arg(
            target_level.lower(), "target_level", self._VALID_TARGET_LEVEL
        )
        if img_size < pet_size:
            raise ValueError("img_size must be greater or equal than pet_size.")
        self._img_size = img_size
        self._pet_size = pet_size
        self._background = verify_str_arg(
            background.lower(), "background", self._VALID_BACKGROUND
        )
        self._hard_cut = hard_cut

        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        # Raw path
        self._root_folder = pathlib.Path(self.root) / "noisybg-oxford-pet"
        self._raw_images_folder = self._root_folder / "raw" / "images"
        self._raw_anns_folder = self._root_folder / "raw" / "annotations"
        # Path to specific data
        self._base_folder = self._root_folder / f"pet_size_{self._pet_size:03}"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"

        # Download
        if download:
            self._download()
            preprocess = False

        # Check if raw data exits in root folder
        if not self._check_exists(self._raw_images_folder, self._raw_anns_folder):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        # Check if pet_size specific data exists and preprocess if not
        if not self._check_exists(self._images_folder, self._anns_folder):
            print("Preprocess data ...", end="")
            self._preprocess()
            print("Done!")

        # Read in data
        self.data = pd.read_csv(self._base_folder / f"{self._split}.csv")

        # Change classes
        if self._target_level == "species":
            self.classes = ["cat", "dog"]
        else:
            self.classes = [
                " ".join(part.title() for part in raw_cls.split("_"))
                for raw_cls, _ in sorted(
                    {
                        (image_id.rsplit("_", 1)[0], label)
                        for image_id, label in zip(self.data["image"], self.data["id"])
                    },
                    key=lambda image_id_and_label: image_id_and_label[1],
                )
            ]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.preload = preload
        if preload:
            self.preload_all_images()

    def _cutout_imgs(self, img, mask):
        """
        Cuts out the pet from the image.
        Args:
            img (torch.Tensor)
            mask (torch.Tensor)
        """
        # get mask with pet+surrounding == 1., rest == 0.
        mask_bool = ((mask * 255) != 2.0).float().squeeze(0)

        # Cut out first and last 2 rows and collums due to labeling error at the border
        img = img[:, 2:-2, 2:-2]
        mask = mask[:, 2:-2, 2:-2]
        mask_bool = mask_bool[2:-2, 2:-2]

        # Get idx of max/min of mask
        where_mask = torch.where(mask_bool == 1.0)
        top = where_mask[0].min()
        bottom = where_mask[0].max()
        left = where_mask[1].min()
        right = where_mask[1].max()

        # pad img + mask and save
        transf = TF.Compose(
            [TF.Resize((self._pet_size, self._pet_size), interpolation=TF.InterpolationMode.NEAREST), TF.ToPILImage()]
        )
        return transf(img[:, top:bottom, left:right]), transf(
            mask[:, top:bottom, left:right]
        )

    def _preprocess(self) -> None:
        # Create target folders
        os.makedirs(self._images_folder)
        os.makedirs(self._anns_folder)

        # read in dataframe
        df_trainval = pd.read_csv(
            self._raw_anns_folder / "trainval.txt",
            sep=" ",
            header=None,
            names=["image", "id", "species", "breed_id"],
        )
        df_test = pd.read_csv(
            self._raw_anns_folder / "test.txt",
            sep=" ",
            header=None,
            names=["image", "id", "species", "breed_id"],
        )

        # drop broken ones
        df_trainval = df_trainval[~df_trainval["image"].isin(self._BROKEN_SEGMENTATION)]
        df_test = df_test[~df_test["image"].isin(self._BROKEN_SEGMENTATION)]
        # Reset index to exclude droped ones from index counting
        df_trainval.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        # Get test and train set
        if self._small_test:
            np.random.seed(1234)
            test_imgs = df_test.groupby("id", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 10))
            )["image"]
            # get new train and test dataset
            df_trainval = pd.concat(
                [df_trainval, df_test[~df_test["image"].isin(test_imgs)]]
            )
            df_test = df_test[df_test["image"].isin(test_imgs)]

        # Rescale labels
        df_trainval.iloc[:, 1:] = df_trainval.iloc[:, 1:] - 1
        df_test.iloc[:, 1:] = df_test.iloc[:, 1:] - 1

        # Save dfs
        df_trainval.to_csv(self._base_folder / "trainval.csv", index=False)
        df_test.to_csv(self._base_folder / "test.csv", index=False)

        # combine Train and test
        df_all = pd.concat([df_trainval, df_test])

        for img_name in df_all["image"]:
            # Load image and segmenation
            img = Image.open(self._raw_images_folder / f"{img_name}.jpg").convert("RGB")
            mask = Image.open(self._raw_anns_folder / "trimaps" / f"{img_name}.png")

            # Transform to tensor
            img = TF.functional.to_tensor(img)
            mask = TF.functional.to_tensor(mask)

            # Cutout image
            img, mask = self._cutout_imgs(img, mask)

            # Save
            img.save(self._images_folder / f"{img_name}.jpg")
            mask.save(self._anns_folder / f"{img_name}.png")

    def _generate_img(self, img, mask, label=None):
        """
        Cuts out the pet from the image.
        Args:
            img (torch.Tensor)
            mask (torch.Tensor)
            label (int or float)
        """
        # Get boolean mask
        if self._hard_cut:
            mask = ((mask * 255) == 1.0).float().squeeze(0)
        else:
            mask = ((mask * 255) != 2.0).float().squeeze(0)

        if self._background == "white":
            img = img * mask + torch.ones_like(img) * (1 - mask)
            img_out = torch.ones(3, self._img_size, self._img_size)
        elif self._background == "big_noise":
            img = img * mask + torch.rand_like(img) * (1 - mask)
            img_out = torch.rand(3, self._img_size, self._img_size)
        else:
            if self._background == "noise":
                label = random.randint(0, 36)
            img = img * mask + (label / 37 + torch.rand_like(img) / 37) * (1 - mask)
            img_out = label / 37 + torch.rand(3, self._img_size, self._img_size) / 37

        # Add image into random position
        if self._img_size > self._pet_size:
            a, b = np.random.randint(self._img_size - self._pet_size, size=(2))
            img_out[:, a: (a + self._pet_size), b: (b + self._pet_size)] = img
        else:
            img_out = img

        # TODO Why do we transform a tensor back to a Pil Image?
        return TF.functional.to_pil_image(img_out)

    def __len__(self) -> int:
        return len(self.data)

    def preload_all_images(self):
        self.image_data = []
        self.mask_data = []
        self.noised_placed_data = []
        for idx in range(len(self.data)):
            self.image_data.append(copy.deepcopy(Image.open(self._images_folder / f"{self.data['image'][idx]}.jpg")))
            self.mask_data.append(copy.deepcopy(Image.open(self._anns_folder / f"{self.data['image'][idx]}.png")))

            # Transform to tensor
            image, mask = TF.functional.to_tensor(self.image_data[-1]), TF.functional.to_tensor(self.mask_data[-1])
            # generate imag
            image = self._generate_img(image, mask, label=self.data["id"][idx])

            self.noised_placed_data.append(image)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self.preload:
            image = self.image_data[idx]
            mask = self.mask_data[idx]
        else:
            # Get image
            image = Image.open(
                self._images_folder / f"{self.data['image'][idx]}.jpg"
            ).convert("RGB")
            # Get mask
            mask = Image.open(self._anns_folder / f"{self.data['image'][idx]}.png")

        if False:
            # Transform to tensor
            image, mask = TF.functional.to_tensor(image), TF.functional.to_tensor(mask)

            # generate imag
            image = self._generate_img(image, mask, label=self.data["id"][idx])
        else:
            image = self.noised_placed_data[idx]
        target = (
            self.data["species"][idx]
            if self._target_level == "species"
            else self.data["id"][idx]
        )

        if self.transform:
            image = self.transform(image)

        return image, target

    @staticmethod
    def _check_exists(img_path, anns_path) -> bool:
        for folder in (img_path, anns_path):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists(self._raw_images_folder, self._raw_anns_folder):
            return

        # Download
        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._root_folder), md5=md5
            )

        # Delete tar files
        os.remove(self._root_folder / "images.tar.gz")
        os.remove(self._root_folder / "annotations.tar.gz")
        # Copy data into raw
        shutil.copytree(self._root_folder / "images", self._raw_images_folder)
        shutil.copytree(self._root_folder / "annotations", self._raw_anns_folder)
        # and then remove Folders
        shutil.rmtree(self._root_folder / "images")
        shutil.rmtree(self._root_folder / "annotations")
