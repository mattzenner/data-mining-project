
import glob # file structure
from PIL import Image # image processing

size = (28, 28) # 28x28 pixels
# im = Image.open(r"D://archive//Ademo100//A1.jpg")
# im.draft('L', size)
# im.load()
# im.show()

images = glob.glob("D://archive//Ademo100//*.jpg") # iterable of all images in Ademo100 with file extension .jpg
for image in images: # for each image in the folder
    print(image)
    with open(image, 'rb') as file: # open image file
        img = Image.open(file)
        img.draft('L', size) # image file mode to grayscale, apply 28x28 size
        img.load() # load image into memory
        img.save(image)

#img = Image.open('image.png').convert('LA')
#img.save('greyscale.png')
