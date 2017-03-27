

# Mirrors (flips) an image, vertically and/or horizontally
def mirror(img, vertical=True):
    out = img.copy()
    if vertical:
        out = cv2.flip(out, 1)

    return out


# Test
def test_transform(imgId, transformation, *trans_args):
    img = X_train[imgId]
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original")
    _ = plt.imshow(img)
    newImg = transformation(img, trans_args)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title("Transformed")
    _ = plt.imshow(newImg)

    plt.show()


# test_transform(6789, mirror)


# distorts the image by a random (but bounded) number of pixels
# in multiple directions
def skew(img, args):
    rows, cols, ch = img.shape
    x1 = random.randint(1, 10)
    y1 = random.randint(5, 10)
    x2 = random.randint(25, 30)
    y2 = y1
    x3 = y2
    y3 = x2
    x4 = random.randint(1, 5)
    y4 = random.randint(1, 5)
    x5 = random.randint(20, 30)
    y5 = random.randint(1, 10)

    pts1 = np.float32([[y1, x1], [y2, x2], [y3, x3]])
    pts2 = np.float32([[x4, y4], [x3, y3], [x5, y5]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))


#test_transform(6789, skew)


# Rotates image by a random but bounded number of degrees
# around a random point close to the center. (max 5px away in each direction)
def rotate(img, args):
    center = (np.array(image.shape[:2]) / 2)
    center[0] += random.randint(-5, 5)
    center[1] += random.randint(-5, 5)
    center = tuple(center)
    angle = random.randint(-25, 25)
    rot = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(img, rot, image.shape[:2], flags=cv2.INTER_LINEAR)


#test_transform(6789, rotate)


# Returns an image on which a gaussian blur filter has been applied
def blur(img, args):
    r = 3
    return cv2.GaussianBlur(img,ksize=(r,r),sigmaX=0)

# test_transform(6789, blur)


# Moves the image
def translation(img, args):
    dx = random.randint(-5,5)
    dy = random.randint(-5,5)
    num_rows, num_cols = img.shape[:2]
    trans_matrix = np.float32([ [1,0,dx], [0,1,dy] ])
    return cv2.warpAffine(img, trans_matrix, (num_cols, num_rows))

# test_transform(6789, translation)


mirrorable = [11, 12, 13, 15, 17, 18, 22, 26, 27, 35]
transforms = [translation, blur, rotate, skew, mirror]
images_in_class = 5000


### This has been executed once and the data is stored in the x_train_augmented and y_train_augmented files

# How many images do we want in each class and how many are missing ?

#classes, counts = np.unique(y_train, return_counts=True)
#for clazz, count in zip(classes, counts):
#    print(str(clazz) + " has " + str(count) + " images, " + str(images_in_class - count) + " to be generated.")
#    y_indices = np.where(y_train == clazz)
#    z = 1 + len(np.where(mirrorable == clazz)[0])
    #for i in range((images_in_class - count)):
        #j = random.randint(0, len(y_indices[0]) - 1)
        #k = y_indices[0][j]
        #img = X_train[k]
        #func = transforms[random.randint(0, len(transforms) - z)]
        #newImg = cv2.cvtColor(func(img, None), cv2.COLOR_BGR2RGB)
        #cv2.imwrite("datasets/augmentation/" + str(clazz) + "/" + str(i) + ".png", newImg)
        #img = cv2.imread("datasets/augmentation/" + str(clazz) + "/" + str(i) + ".png")
        #X_train = np.append(X_train, img)
        #y_train = np.append(y_train, clazz)
        #if(i%50 == 0):
        #    print(i)

    #pickle.dump(X_train, open("x_train_augmented_"+str(clazz)+".p", "wb"))

