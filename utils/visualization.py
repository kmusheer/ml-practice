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
    # plt.savefig("roc_curve.png")
    plt.show()


def plot_residuals(y_test,y_pred, name) :
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0)
    plt.title(f"Residual Plot - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'r--')

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()