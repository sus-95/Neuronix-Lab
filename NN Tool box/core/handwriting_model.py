import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
IMG_SIZE = 28

def center_image(img):
    """Centers the content of a grayscale PIL image."""
    # Get bounding box
    bbox = ImageOps.invert(img.point(lambda x: 0 if x > 20 else 255)).getbbox()
    if not bbox:
        return img
    
    # Crop to bounding box
    char_img = img.crop(bbox)
    
    # Calculate new size to fit in 20x20 while maintaining aspect ratio
    w, h = char_img.size
    ratio = min(20/w, 20/h)
    new_size = (int(w*ratio), int(h*ratio))
    char_img = char_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Paste onto a new 28x28 image
    centered_img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    centered_img.paste(char_img, ((28-new_size[0])//2, (28-new_size[1])//2))
    return centered_img

def generate_synthetic_data(samples_per_class=200):
    """Generates robust synthetic A-Z images using PIL."""
    X = []
    y = []
    
    # Try to find fonts
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf"
    ]
    fonts = []
    for path in font_paths:
        if os.path.exists(path):
            fonts.append(ImageFont.truetype(path, 22))
    if not fonts:
        fonts.append(ImageFont.load_default())

    for i, char in enumerate(CLASSES):
        for _ in range(samples_per_class):
            # Create a larger image first for better rotation
            img = Image.new('L', (40, 40), color=0)
            draw = ImageDraw.Draw(img)
            
            font = np.random.choice(fonts)
            
            # Randomize position slightly before centering
            offset_x = np.random.randint(5, 15)
            offset_y = np.random.randint(5, 15)
            
            draw.text((offset_x, offset_y), char, fill=255, font=font)
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle)
            
            # Center and resize to 28x28
            img = center_image(img)
            
            # Add some noise/blur
            data = np.array(img).astype('float32')
            noise = np.random.normal(0, 5, (IMG_SIZE, IMG_SIZE))
            data = np.clip(data + noise, 0, 255)
            
            X.append(data)
            y.append(i)
            
    X = np.array(X).astype('float32') / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    
    return X, y

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_hopfield_patterns(size=28):
    """Generates clean centered patterns for A-Z of a specific size."""
    patterns = []
    # Use a clean bold font for patterns
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf"
    
    try:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size - 6 if size > 10 else size - 2)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    for char in CLASSES:
        img = Image.new('L', (size + 10, size + 10), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), char, fill=255, font=font)
        
        # Center and resize to target size
        bbox = ImageOps.invert(img.point(lambda x: 0 if x > 20 else 255)).getbbox()
        if bbox:
            char_img = img.crop(bbox)
            w, h = char_img.size
            ratio = min((size-4)/w, (size-4)/h)
            new_size = (max(1, int(w*ratio)), max(1, int(h*ratio)))
            char_img = char_img.resize(new_size, Image.Resampling.LANCZOS)
            
            final_img = Image.new('L', (size, size), color=0)
            final_img.paste(char_img, ((size-new_size[0])//2, (size-new_size[1])//2))
            img = final_img
        else:
            img = img.resize((size, size))
        
        # Convert to -1, 1 for Hopfield
        data = np.array(img)
        pattern = np.where(data > 100, 1, -1).flatten()
        patterns.append(pattern)
        
    return np.array(patterns)

def train_and_save(model_path='core/letter_cnn.h5'):
    print("Generating robust synthetic data...")
    X, y = generate_synthetic_data(samples_per_class=300)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    print("Building improved model...")
    model = build_cnn()
    
    print("Training...")
    # Use some simple augmentation during training
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    model.fit(datagen.flow(X, y, batch_size=32), epochs=15, verbose=1)
    
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    return model
