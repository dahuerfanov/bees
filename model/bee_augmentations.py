import imgaug.augmenters as iaa
import torchvision

from model.bee_center_sample import BeeCenterSample
from model.transforms.image import ImageAugmentation
from model.transforms.sample import CategoryIdToClass, ComposeSample


def get_transforms(norm_mean, norm_std, valid_ids, max_objs, kernel_px):
    train_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential([
                    iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 500}),
                    iaa.Sequential([
                            iaa.Fliplr(0.5),
                            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                            iaa.LinearContrast((0.75, 1.5)),
                            iaa.AdditiveGaussianNoise(
                                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                            ),
                            iaa.Multiply((0.8, 1.2), per_channel=0.1),
                            iaa.Affine(
                                scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                                translate_percent={
                                    "x": (-0.2, 0.2),
                                    "y": (-0.2, 0.2),
                                },
                                rotate=(-5, 5),
                                shear=(-3, 3),
                            ),
                    ], random_order=True),
                    iaa.PadToFixedSize(width=500, height=500),
                    iaa.CropToFixedSize(width=500, height=500),
                    iaa.PadToFixedSize(width=512, height=512, position="center"),
                ]),
                torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std, inplace=True),
                ]),
            ),
            CategoryIdToClass(valid_ids),
            BeeCenterSample(num_classes=1, max_objects=max_objs, kernel_px=kernel_px),
        ]
    )

    valid_transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential([
                    iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 500}),
                    iaa.PadToFixedSize(width=512, height=512, position="center"),
                ]),
                torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std, inplace=True),
                ]),
            ),
            CategoryIdToClass(valid_ids),
            BeeCenterSample(num_classes=1, max_objects=max_objs, kernel_px=32),
        ]
    )

    test_transform = ImageAugmentation(
                iaa.Sequential([
                    iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 500}),
                    iaa.PadToFixedSize(width=512, height=512, position="center"),
                ]),
                torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std, inplace=True),
                ]),
            )

    return train_transform, valid_transform, test_transform