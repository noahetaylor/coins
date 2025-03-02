
import matplotlib.pyplot as plt

def chartTraining(trainingResults):
    acc = trainingResults.history['accuracy']
    val_acc = trainingResults.history['val_accuracy']
    loss = trainingResults.history['loss']
    val_loss = trainingResults.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()