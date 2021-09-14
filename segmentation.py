
import albumentations as A
from PIL import Image
import numpy as np
import cv2


image = cv2.imread("images/car.jpg")
mask = np.array(Image.open("images/car_mask.gif").convert("L"))

#  imread() won't read any .gif there is no codec for this (license problem)
# cap = cv2.VideoCapture("images/car_mask.gif")
# ret, mask = cap.read()
# cap.release()



# print(image.shape)
# print(mask.shape)

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
    print(i)
    augmentations = transform(image=image, mask=mask)
    aug_img = augmentations["image"]
    aug_mask = augmentations["mask"]
    cv2.imwrite(f"output/segmentation/image/out{i}.jpg", aug_img)
    cv2.imwrite(f"output/segmentation/mask/out{i}.jpg", aug_mask)
