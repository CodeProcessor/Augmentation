
import albumentations as A
from PIL import Image
import numpy as np
import cv2

# image = Image.open("images/girl.jpg")
image = cv2.imread("images/girl.jpg")

transform = A.Compose(
    [
        A.Resize(width=100, height=100),
        A.RandomCrop(width=100, height=100),
        A.Rotate(limit=30, p=0.6, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.transforms.ColorJitter(p=0.5)
            ], p=0.8
        )
    ]
)

for i in range(20):
    augmentations = transform(image=image)
    aug_img = augmentations["image"]
    cv2.imwrite(f"output/classification/out{i}.jpg", aug_img)
