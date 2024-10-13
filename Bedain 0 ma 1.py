import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import cv2
import os

# Fungsi untuk memuat gambar dan meratakannya
def load_images_from_folders(folders):
    images = []
    for folder in folders:
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Mengonversi ke grayscale
            img = cv2.resize(img, (30, 30),)  # Mereskalakan ukuran ke 30x30
            images.append(np.array(img).flatten())  # Meratakan dan menambahkan ke list
    return images

# Path relatif ke folder 0 dan 1
folders = ["./digit_dataset2/0", "./digit_dataset2/1"]


# Memuat gambar dari kedua folder
flattened_images = load_images_from_folders(folders)

# Mengubah menjadi array NumPy
X = np.array(flattened_images)
y = np.vstack((np.zeros((190,1)), np.ones((190,1))))

print("Matriks X dengan ukuran {}:".format(X.shape))
print(X)
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))

# m, n = X.shape

# fig, axes = plt.subplots(8,8, figsize=(8,8))
# fig.tight_layout(pad=0.1)

# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
    
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((30,30))
    
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
    
#     # Display the label above the image
#     ax.set_title(y[random_index,0])
#     ax.set_axis_off()
    
# plt.show()

model = Sequential(                      
    [                                   
        tf.keras.Input(shape=(900,)),    # specify input size (optional)
        Dense(25, activation='relu'), 
        Dense(15, activation='relu'), 
        Dense(1,  activation='sigmoid')  
    ], name = "my_model"                                    
)
model.summary()
L1_num_params = 900 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )
[layer1, layer2, layer3] = model.layers

#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
print(f"W1 bernilai{W1}\n")

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=20
)
print(f"W1 bernilai{W1}\n")
prediction = model.predict(X[0].reshape(1,900))  # a zero
print(f" predicting a zero: {prediction}")
prediction = model.predict(X[300].reshape(1,900))  # a one
print(f" predicting a one:  {prediction}")

new_image_path= './Test'
new_images=[]
for filename in os.listdir(new_image_path):
    img_new_path = os.path.join(new_image_path, filename)
    img_new = cv2.imread(img_new_path, cv2.IMREAD_GRAYSCALE)  # Mengonversi ke grayscale
    img_new = cv2.resize(img_new, (30, 30),)  # Mereskalakan ukuran ke 30x30
    new_images.append(np.array(img_new).flatten()) 

Test = np.array(new_images)
print("String Test adalah" + str(Test.shape))
predictions = model.predict(Test)
m_test,n_test= Test.shape
print("m_ttest="+str(m_test))
fig, axes = plt.subplots(5,4, figsize=(5,6))
fig.tight_layout(pad=3)

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m_test)
    
    # Select rows corresponding to the random indices and
    # Ambil gambar dari Test menggunakan indeks acak dan reshape
    Test_random_reshaped = Test[random_index].reshape((30, 30))
    
    # Tentukan kelas prediksi untuk gambar ini
    predicted_class = 1 if predictions[random_index] >= 0.5 else 0
    # Display the image
    ax.imshow(Test_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(f"Pred: {predicted_class}")
    ax.set_axis_off()
    fig.tight_layout()
    
plt.show()
