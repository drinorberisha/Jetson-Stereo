import cv2
import numpy as np
import os

def enhance_image(image_path):
    os.makedirs('../images/enhanced', exist_ok=True)
    print(f"Enhancing image: {image_path}")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray)
    output_path = f'../images/enhanced/enhanced_{image_path.split("/")[-1]}'
    return enhanced_img, output_path

def main():
    print("Loading image for enhancement...")
    enhanced_image, output_path = enhance_image('../images/data/imgs/rightcamera/Im_R_5.png')
    if enhanced_image is not None:
        cv2.imwrite(output_path, enhanced_image)
        cv2.imshow('Enhanced Image', enhanced_image)
        print(f"Press 'q' to close the window and exit.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print(f"Image enhancement complete. Enhanced image saved as {output_path}.")
    else:
        print("Image enhancement failed.")

if __name__ == "__main__":
    main()
