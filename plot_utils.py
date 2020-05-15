import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def kth_confusion_matrix(model, dataset):
    onehot_targets = [sample[1].numpy() for sample in dataset]
    onehot_targets = np.concatenate(onehot_targets, axis=0)
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
