import tensorflow as tf
import tensorflow_datasets as tfds

# Załaduj zbiór COCO 2017 z TFDS (wersja klasyfikacji obrazów)
dataset, info = tfds.load('coco/2017', split='train', with_info=True)

num_classes = info.features['objects']['label'].num_classes
print(f'Liczba klas: {num_classes}')

# Funkcja do przygotowania danych (skalowanie, batching)
def preprocess(sample):
    # Obraz do float32 i normalizacja [0,1]
    image = tf.image.resize(sample['image'], (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    # Pobierz etykietę - tu bierzemy pierwszą etykietę z obiektów (może być wiele obiektów)
    label = sample['objects']['label'][0]
    label = tf.one_hot(label, depth=num_classes)

    return image, label

batch_size = 32
train_ds = dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Budowa modelu transfer learning z MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie (np. 3 epoki dla testu)
model.fit(train_ds, epochs=3)
