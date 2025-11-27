"""
Convert legacy .h5 model to SavedModel format
Run this once to convert your model, then update MODEL_PATH in app.py
"""
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the old model
print("Loading model from koa_enhanced_model.h5...")
try:
    model = load_model('koa_enhanced_model.h5', compile=False)
    print("Model loaded successfully!")
    
    # Save in new format
    print("Saving in SavedModel format...")
    model.save('koa_enhanced_model_saved', save_format='tf')
    print("✅ Model saved to 'koa_enhanced_model_saved' directory")
    print("\nUpdate your app.py:")
    print("MODEL_PATH = 'koa_enhanced_model_saved'")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Load weights only
    try:
        # You'll need to reconstruct the model architecture
        print("This requires reconstructing the model architecture.")
        print("Do you have the training notebook/script?")
    except Exception as e2:
        print(f"❌ Alternative failed: {e2}")
