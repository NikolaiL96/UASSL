from .utils import one_channel_transforms, simclr_transforms, TwoTransform, Transform
import torchvision.transforms as T

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class CURRICULUM_augmentations(TwoTransform):
    def __init__(self, image_size, magnitude=1.0, normalize=IMAGENET_NORM):
        trans1 = simclr_transforms(image_size, normalize=normalize, magnitude=magnitude)
        trans2 = simclr_transforms(image_size, p_blur=0.1, p_solarize=0.2, normalize=normalize, magnitude=magnitude)

        super().__init__(trans1, trans2)


class BYOL_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = simclr_transforms(image_size, normalize=normalize)
        trans2 = simclr_transforms(image_size, p_blur=0.1, p_solarize=0.2, normalize=normalize)

        super().__init__(trans1, trans2)


class Supervised_augmentations(Transform):
    def __init__(self, image_size, normalize):
        train_tfms = T.Compose([T.RandomCrop(image_size, padding=4, padding_mode='reflect'),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(*normalize)])
        super().__init__(train_tfms)


class SimSiam_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = simclr_transforms(image_size,
                                   jitter=(0.4, 0.4, 0.4, 0.1),
                                   p_blur=0.5,
                                   normalize=normalize)

        super().__init__(trans1)


class VICReg_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = simclr_transforms(image_size,
                                   p_blur=0.5,
                                   p_solarize=0.1,
                                   normalize=normalize)

        super().__init__(trans1)


class OneChannel_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=None):
        trans1 = one_channel_transforms(image_size, normalize=normalize)
        trans2 = one_channel_transforms(image_size, p_solarize=0.2, normalize=normalize)

        super().__init__(trans1, trans2)
