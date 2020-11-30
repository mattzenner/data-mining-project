import glob # library for navigating folders and files
from PIL import Image # image processing
import pandas as pd

size = (28, 28) # 28x28 pixels, 28*28=784 total pixels
cols = ['label'] + ['pixel{}'.format(i) for i in range(1, 785)] # list of column names: label, pixel1, pixel2, pixel3 ... pizel784
signs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space', 'nothing'] # list of folder names for each sign

def sign_to_label(sign): # input sign from list of signs, output number 1-29 (letters 1-26, +3 for del, space, nothing)
    return {s: i for s, i in zip(signs, range(1, 30))}[sign]

#print(cols)
df = pd.DataFrame()

for sign in signs:
    images = glob.glob("D://archive//train_demo//{}//*.jpg".format(sign)) # for each folder
    label = sign_to_label(sign) # label is number corresponding to letter or extra 3 classes
    for image in images: # for each image in the signs' folder
        #print(image)
        with open(image, 'rb') as file: # open the image in read binary mode (binary because image file)
            img = Image.open(file) # open image
            img.draft('L', img.size) # make the image grayscale(L) and size
            img.load() # load image into memory
            img = img.resize(size) # ensure the correct size
            # print(type(img.getdata()))
            data = [label] + list(img.getdata()) # create the row with the label and 784 pixel values
            # print(len(data))
            # print(data)
            df = df.append(pd.Series(data), ignore_index=True) # append the row to the dataframe, ignore_index is True because False throws error
            # print(len(df))
            # img.save(image)
    print('Sign completed: ', sign) # print the completed sign to give programmer sense of progress

print(df) # print before and after renaming columns
df.columns = cols
print(df)

with open('D://archive//test_A.csv', mode='w') as file: # save dataframe to csv
    df.to_csv(path_or_buf=file, index=False)

# img = Image.open(r"D://archive//Ademo100//A1.jpg")
# img.load()
# print('Size: ', img.size)
# print('Bands: ', img.getbands())
# print('Pixel data: ', list(img.getdata()))
# print('Number of pixels: ', len(list(img.getdata())))
# img = img.resize(size)
# print('Size: ', img.size)
# print('Bands: ', img.getbands())
# print('Pixel data: ', list(img.getdata()))
# print('Number of pixels: ', len(list(img.getdata())))
