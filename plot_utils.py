import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def kth_confusion_matrix(model, dataset, onehot_encoded=False):
    targets = [sample[1].numpy() for sample in dataset]
    if onehot_encoded:
        targets = np.concatenate(targets, axis=0)
        targets = np.argmax(onehot_targets, axis=1)

    onehot_preds = model.predict(dataset)
    preds = np.argmax(onehot_preds, axis=1)
    conf = confusion_matrix(targets, preds)

    ax = plt.subplot()
    sns.heatmap(conf, annot=True, ax=ax, cmap=sns.cubehelix_palette(8), fmt='g',
        xticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'],
        yticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

def plot_history(history):
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Training accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation accuracy")
    plt.title("Accuracy during training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
