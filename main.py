import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

print("=" * 50)
print("🌱 Crop Disease Detection - Training")
print("=" * 50)

# সম্পূর্ণ পাথ ব্যবহার করা (স্পেস সমস্যা এড়াতে)
dataset_path = r"C:\Users\Nidhi\OneDrive\Desktop\Crop_derection with ai\dataset"

# চেক করা কয়টা ছবি আছে
healthy_path = os.path.join(dataset_path, 'healthy')
diseased_path = os.path.join(dataset_path, 'diseased')

h_count = len([f for f in os.listdir(healthy_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
d_count = len([f for f in os.listdir(diseased_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"✅ healthy ফোল্ডারে: {h_count} টা ছবি")
print(f"✅ diseased ফোল্ডারে: {d_count} টা ছবি")
print(f"📊 মোট: {h_count + d_count} টা ছবি")

# ইমেজ জেনারেটর (validation ছাড়া)
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=2,
    class_mode='binary'
)

print(f"\n🔍 TensorFlow খুঁজে পেয়েছে: {train_generator.samples} টা ছবি")

# মডেল তৈরি
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ট্রেইনিং
print("\n🚀 Training শুরু...")
model.fit(train_generator, epochs=10)

# সেভ করা
model_dir = r"C:\Users\Nidhi\OneDrive\Desktop\Crop_derection with ai\model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
model.save(os.path.join(model_dir, 'crop_disease_model.h5'))
print("\n✅ সফল! মডেল সেভ হয়েছে!")