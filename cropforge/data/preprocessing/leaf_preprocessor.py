import os
import cv2
import numpy as np


class LeafPreprocessor:

    def __init__(self, image_path, target_size=(640, 640), use_masking=True):
        self.target_size = target_size
        self.image_path = image_path
        self.use_masking = use_masking

        # Initialize image variables
        self.image = None
        self.load_image()

    def load_image(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found at {self.image_path}")

        img = cv2.imread(self.image_path)
        if img is not None:
            self.image = img
        else:
            raise ValueError(
                f"File exists but cv2 could not decode image: {self.image_path}"
            )

    def validate_image(self):
        if self.image is None:
            raise ValueError("No image loaded to validate.")

        # Ensure it is a 3-channel BGR image
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            return True
        else:
            raise TypeError(f"Invalid image format: {self.image_path}")

    def resize_image(self, img, width, height):
        return cv2.resize(img, (width, height))

    def denoise_image(self, img):
        # Bilateral filter preserves sharp leaf edges while smoothing textures
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def convert_to_hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def normalize_image(self, img):
        # Good practice: explicitly cast to float32 if normalizing for deep learning models [0, 1]
        # Or keep it uint8 [0, 255] depending on your network's requirement
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    def create_leaf_mask(self, hsv_image, lower_bound, upper_bound):
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def preprocess(self):
        # 1. Validate raw data
        self.validate_image()
        original_raw = self.image.copy()

        # 2. Base Pipeline Sequence
        resized_image = self.resize_image(
            original_raw, self.target_size[0], self.target_size[1]
        )
        denoised_image = self.denoise_image(resized_image)

        # 3. Masking Pipeline (Uses clean, un-normalized HSV data)
        masked_image = None
        leaf_mask = None
        hsv_image = self.convert_to_hsv(denoised_image)

        if self.use_masking:
            # Standard HSV bounds for green plant matter
            lower_bound = np.array([25, 40, 40])
            upper_bound = np.array([85, 255, 255])

            leaf_mask = self.create_leaf_mask(
                hsv_image, lower_bound, upper_bound
            )
            # Apply mask to the denoised BGR image
            masked_image = cv2.bitwise_and(
                denoised_image, denoised_image, mask=leaf_mask
            )

        # 4. Final Normalization (Apply normalization on the final outputs if needed)
        # If masking was applied, we normalize the isolated leaf image
        base_for_normalization = (
            masked_image if self.use_masking else denoised_image
        )
        normalized_image = self.normalize_image(base_for_normalization)

        return {
            "original_image": original_raw,
            "resized_image": resized_image,
            "denoised_image": denoised_image,
            "hsv_image": hsv_image,
            "leaf_mask": leaf_mask,
            "masked_image": masked_image,
            "normalized_final": normalized_image,
        }


if __name__ == "__main__":
    # Example usage:
    # preprocessor = LeafPreprocessor("path_to_leaf.jpg")
    # results = preprocessor.preprocess()
    pass