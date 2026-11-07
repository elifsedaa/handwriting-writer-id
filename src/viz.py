"""Görselleştirme yardımcıları (Gün 9)."""
import matplotlib.pyplot as plt
import numpy as np


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def plot_predictions(train_data, train_labels, test_data, test_labels,
                     predictions=None, title="Eğitim, test ve tahmin", path=None):
    """Eğitim (mavi), test (yeşil) ve varsa tahmin (kırmızı) noktalarını çizer."""
    plt.figure(figsize=(8, 5))
    plt.scatter(_np(train_data), _np(train_labels), c="b", s=6, label="Eğitim verisi")
    plt.scatter(_np(test_data), _np(test_labels), c="g", s=10, label="Test verisi")
    if predictions is not None:
        plt.scatter(_np(test_data), _np(predictions), c="r", s=10, label="Tahminler")
    plt.legend(prop={"size": 11})
    plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_decision_boundary(model, X, y, path=None, title="Karar sınırı", device="cpu"):
    """2 özellikli veride modelin karar sınırını çizer (Gün 17-18).

    Tek çıktılı (ikili) modelde sigmoid, çok çıktılıda softmax+argmax kullanılır.
    """
    import torch

    X_np = _np(X)
    pad = 0.1
    x_min, x_max = X_np[:, 0].min() - pad, X_np[:, 0].max() + pad
    y_min, y_max = X_np[:, 1].min() - pad, X_np[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    grid = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float().to(device)

    model.to(device).eval()
    with torch.inference_mode():
        logits = model(grid)
    if logits.shape[1] > 1:
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
    else:
        pred = torch.round(torch.sigmoid(logits)).squeeze(1)
    zz = _np(pred).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=_np(y), s=8, cmap=plt.cm.RdYlBu, edgecolors="k", linewidths=0.2)
    plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max()); plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_loss_curves(history: dict, path=None, title="Kayıp eğrileri"):
    """history: {'epoch': [...], 'train_loss': [...], 'test_loss': [...]} sözlüğünü çizer."""
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["test_loss"], label="test")
    plt.xlabel("epoch"); plt.ylabel("kayıp"); plt.title(title); plt.legend()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
