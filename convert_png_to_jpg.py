from PIL import Image
import glob

count = 0
images = glob.glob("training_set/A/*.png") # images becomes an iterable of files in the given path
for image in images: # iterate through each image in images
    with open(image, 'rb') as file: # open the current image in read-binary mode as file
        img = Image.open(file) # create a PIL Image object from file
        img = img.convert('RGB') # convert to RGB mode
        img.load() # call load function to 'commit' changes to image
        img.save('training_set/A/A_convert{}.jpg'.format(str(count))) # save the file in jpg at given path
    count += 1 # provide a counter to give each saved jpg a unique file name by its number
print(count)
print('Done!')

