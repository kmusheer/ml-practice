import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0,1], [0,1], '--', label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()
