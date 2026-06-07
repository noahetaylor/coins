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

    numImages = 0
    numErrors = 0
    directoryPath = Path(pathToTestImages)
    categories = [f for f in directoryPath.iterdir() if f.is_dir()]
    categories.sort()
    for category in categories:
        imageDir = os.path.join(directoryPath, category)
        imageDirPath = Path(imageDir)
        files = [f for f in imageDirPath.iterdir() if f.is_file()]
       
        for file in files:
            numImages = numImages + 1
            img = tf.keras.utils.load_img(file, target_size=(imageSize, imageSize), keep_aspect_ratio=True)
            x = tf.keras.utils.img_to_array(img)
            x = np.explaband_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            predictionIndex = 0
            if len(classes[0]) == 1:
                if classes[0] > 0.5:
                    predictionIndex = 1
            else:
                predictionIndex = np.argmax(classes[0])
            # prediction = np.argmax(classes)
            # print(f"{file} - {prediction}")
            # bestMatchIndex = 0
            # bestProbability = 0.0
            # for i, probability in enumerate(classes[0]):
            #     if probability > bestProbability:
            #         bestMatchIndex = i
            #         bestProbability = probability
            if category != categories[predictionIndex]:
                numErrors = numErrors + 1
                print(f"[{category.name}]'{file.name}' => {categories[predictionIndex].name}")
                outFile = categories[predictionIndex].name + "-" + file.name
                dst = os.path.join(outputFolder, outFile)
                shutil.copy(file, dst)
    percentCorrect = float(numImages - numErrors) / float(numImages)
    print("Accuracy=", percentCorrect)



def findHierarchyMiscategorizedImages(headsVsTailsModel, headsModel, tailsModel, pathToTestFolder, outputFolder, imageSize):
    shutil.rmtree(outputFolder, True)

    numImages = 0
    numErrors = 0
    broadCategories = ["Heads", "Tails"]
    fineModels = [headsModel, tailsModel]
    for broadCategory in broadCategories:
        broadCategoryDir = os.path.join(pathToTestFolder, broadCategory)
        broadCategoryPath = Path(broadCategoryDir)
        broadCategoryOutputFolder =  os.path.join(outputFolder, broadCategory)
        create_directory(broadCategoryOutputFolder)
        fineCategories = [f for f in broadCategoryPath.iterdir() if f.is_dir()]
        fineCategories = sorted(fineCategories, key=lambda x: x.name)
        for fineCategory in fineCategories:
            imageDir = os.path.join(broadCategoryDir, fineCategory)
            fineCategoryOutputFolder =  os.path.join(broadCategoryOutputFolder, fineCategory.name)
            create_directory(fineCategoryOutputFolder)
            imageDirPath = Path(imageDir)
            files = [f for f in imageDirPath.iterdir() if f.is_file()]
        
            for file in files:
                numImages = numImages + 1
                img = tf.keras.utils.load_img(file, target_size=(imageSize, imageSize))
                x = tf.keras.utils.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                images = np.vstack([x])
                classes = headsVsTailsModel.predict(images, batch_size=10)
                predictedBroadIndex = getBestMatch(classes[0])
                predictedBroadCategory = broadCategories[predictedBroadIndex]
                if broadCategory != predictedBroadCategory:
                    numErrors = numErrors + 1
                    print(f"[{broadCategory}]'{file.name}' => {predictedBroadCategory}")
                    outFile = predictedBroadCategory + "--" + file.name
                    dst = os.path.join(broadCategoryOutputFolder, outFile)
                    shutil.copy(file, dst)
                else:
                    classes = fineModels[predictedBroadIndex].predict(images, batch_size=10)
                    predictedFineIndex = getBestMatch(classes[0])
                    predictedFineCategory = fineCategories[predictedFineIndex]
                    if fineCategory != predictedFineCategory:
                        numErrors = numErrors + 1
                        print(f"[{broadCategory}-{fineCategory.name}]'{file.name}' => {predictedBroadCategory}-{predictedFineCategory.name}")
                        outFile = predictedBroadCategory + "-" + predictedFineCategory.name + "--" + file.name
                        dst = os.path.join(fineCategoryOutputFolder, outFile)
                        shutil.copy(file, dst)
    percentCorrect = float(numImages - numErrors) / float(numImages)
    print("Accuracy=", percentCorrect)


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