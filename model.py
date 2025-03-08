import tensorflow as tf

def load_ocr_model(model_path):
    """Loads the trained OCR model."""
    return tf.keras.models.load_model(model_path)

# Example usage
if __name__ == "__main__":
    model_path = r"C:\Users\Omar Mohamed\OneDrive\Desktop\ThesisTextTranslation\ocr_character_model.h5"
    ocr_model = load_ocr_model(model_path)
    print("OCR model loaded successfully!")
