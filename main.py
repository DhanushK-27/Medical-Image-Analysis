import os
import cv2
import matplotlib.pyplot as plt

from medical_image import load_image, process_image


IMAGE_PATH = "dataset/sample_xray.png"
OUTPUT_DIR = "results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = load_image(IMAGE_PATH)
    outputs = process_image(image)

    # Display original and processed images.
    plt.figure(figsize=(14, 8))

    for index, (name, result) in enumerate(outputs.items(), start=1):
        plt.subplot(2, 4, index)
        plt.imshow(result, cmap="gray")
        plt.title(name)
        plt.axis("off")

        if name != "Original":
            filename = name.lower().replace(" ", "_") + ".png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, result)

    plt.tight_layout()
    plt.show()

    print("Image processing completed successfully.")
    print(f"Processed images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()