from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import shutil

def findMiscategorizedImages(model, pathToTestImages, outputFolder, imageSize=150):
    """
    This function takes a trained AI model to categorize the results
    of the test images.  The subfolders in pathToTestImages should 
    be the name of the tags for the images. 

    Parameters:
    model: The trained AI model
    pathToTestImages: The path to the images to categorize

    Returns:
    A list of filenames that were miscategorized
    """
    shutil.rmtree(outputFolder, True)
    create_directory(outputFolder)
    directoryPath = Path(pathToTestImages)
    categories = [f for f in directoryPath.iterdir() if f.is_dir()]
    for category in categories:
        imageDir = os.path.join(directoryPath, category)
        imageDirPath = Path(imageDir)
        files = [f for f in imageDirPath.iterdir() if f.is_file()]
       
        for file in files:
            img = tf.keras.utils.load_img(file, target_size=(imageSize, imageSize))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            bestMatchIndex = 0
            bestProbability = 0.0
            for i, probability in enumerate(classes[0]):
                if probability > bestProbability:
                    bestMatchIndex = i
                    bestProbability = probability
            if category != categories[bestMatchIndex]:
                print(f"[{category.name}]'{file.name}' => {categories[bestMatchIndex].name}")
                outFile = categories[bestMatchIndex].name + "-" + file.name
                dst = os.path.join(outputFolder, outFile)
                shutil.copy(file, dst)


def findHierarchyMiscategorizedImages(model, pathToTestImages, outputFolder, imageSize=150):
    """
    This function takes a trained AI model to categorize the results
    of the test images.  The subfolders in pathToTestImages should 
    be the name of the tags for the images. 

    Parameters:
    model: The trained AI model
    pathToTestImages: The path to the images to categorize

    Returns:
    A list of filenames that were miscategorized
    """
    shutil.rmtree(outputFolder, True)
    create_directory(outputFolder)
    directoryPath = Path(pathToTestImages)
    broadCategories = [f for f in directoryPath.iterdir() if f.is_dir()]
    for broadCategory in broadCategories:
        broadCategoryDir = os.path.join(directoryPath, broadCategory)
        broadCategoryPath = Path(broadCategoryDir)
        fineCategories = [f for f in broadCategoryPath.iterdir() if f.is_dir()]
        for fineCategory in fineCategories:
            imageDir = os.path.join(broadCategoryDir, fineCategory)
            print(imageDir)
            imageDirPath = Path(imageDir)
            files = [f for f in imageDirPath.iterdir() if f.is_file()]
        
            for file in files:
                print(file.name)
                img = tf.keras.utils.load_img(file, target_size=(imageSize, imageSize))
                x = tf.keras.utils.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                images = np.vstack([x])
                classes = model.predict(images, batch_size=10)

                for b in range(len(broadCategories)):
                    predictedBroadIndex = getBestMatch(classes[b][0])
                    predictedBroadCategory = broadCategories[predictedBroadIndex]
                    for f in range(len(fineCategories)):
                        predictedFineIndex = getBestMatch(classes[f][0])
                        predictedFineCategory = fineCategories[predictedFineIndex]
                        if fineCategory != predictedFineCategory:
                            print(f"[{fineCategory.name}]'{file.name}' => {predictedBroadCategory.name}-{predictedFineCategory.name}")
                            outFile = predictedFineCategory.name + "-" + file.name
                            dst = os.path.join(outputFolder, outFile)
                            shutil.copy(file, dst)

def getBestMatch(probabilities):
    bestMatchIndex = 0
    bestProbability = 0.0
    for i, probability in enumerate(probabilities):
        if probability > bestProbability:
            bestMatchIndex = i
            bestProbability = probability
    return bestMatchIndex
  

def create_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
        except OSError as e:
            print(f"An error occurred: {e}")
    else:
        print(f"Directory '{path}' already exists.")