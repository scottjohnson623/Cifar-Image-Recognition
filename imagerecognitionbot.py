import cv2
import tensorflow as tf

CATEGORIES = [0, 1,2,3,4,5,6,7,8,9]


def prepare(filepath):
    IMG_SIZE = 32  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return new_array

model = tf.keras.models.load_model('cifarimagerecognition.model')
val = model.predict_classes(prepare("airplanetest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("airplanetest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("cartest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("cartest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("birdtest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("birdtest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("cattest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
# val = model.predict_classes(prepare("cattest2.jpeg"))  # will be a list in a list.
# print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("deertest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("deertest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
# val = model.predict_classes(prepare("dogtest.jpeg"))  # will be a list in a list.
# print(CATEGORIES[val[0]])
# val = model.predict_classes(prepare("dogtest2.jpeg"))  # will be a list in a list.
# print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("frogtest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("frogtest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("horsetest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("horsetest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("shiptest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("shiptest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("semitest.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])
val = model.predict_classes(prepare("semitest2.jpeg"))  # will be a list in a list.
print(CATEGORIES[val[0]])