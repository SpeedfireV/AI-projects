import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


class CustomGaussianNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            if len(X_c) > 0:
                self._mean[idx, :] = X_c.mean(axis=0)
                self._var[idx, :] = X_c.var(axis=0)
                self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self._priors[idx] + 1e-11)
            mean = self._mean[idx]
            var = self._var[idx] + 1e-9
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mean) ** 2 / var
            likelihood = np.sum(log_pdf)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]



def plot_feature_distribution(X_data, y_data, feat_idx, feature_names, classes, filename):
    plt.figure(figsize=(10, 6))
    name = feature_names[feat_idx]
    for q in classes:
        subset = X_data[y_data == q, feat_idx]
        if len(subset) > 0:
            plt.hist(subset, bins=20, density=True, alpha=0.3, label=f'Jakość {q}')
    plt.title(f'Rozkład cechy: {name}')
    plt.xlabel('Wartość')
    plt.ylabel('Gęstość')
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, title, filename, cmap):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=cmap, alpha=0.7)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Przewidziana jakość')
    plt.ylabel('Rzeczywista jakość')
    plt.title(title, pad=20)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def calculate_per_class_accuracy(y_true, y_pred, classes):
    accuracies = []
    for c in classes:
        mask = (y_true == c)
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            accuracies.append(acc * 100)
        else:
            accuracies.append(0.0)
    return accuracies


red_df = pd.read_csv('winequality-red.csv', sep=';')
white_df = pd.read_csv('winequality-white.csv', sep=';')

red_df['wine_type'] = 'red'
white_df['wine_type'] = 'white'

df = pd.concat([red_df, white_df], ignore_index=True)

X = df.drop(['quality', 'wine_type'], axis=1).values
y = df['quality'].values
wine_types = df['wine_type'].values
feature_names = df.drop(['quality', 'wine_type'], axis=1).columns

X_train, X_val, y_train, y_val, type_train, type_val = train_test_split(
    X, y, wine_types, test_size=0.3, random_state=42
)

model = CustomGaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

all_classes = sorted(np.unique(y))
plt.style.use('ggplot')

plot_feature_distribution(X_train, y_train, 10, feature_names, all_classes, 'rozklad_alcohol.png')
plot_feature_distribution(X_train, y_train, 1, feature_names, all_classes, 'rozklad_volatile_acidity.png')

acc_red = calculate_per_class_accuracy(y_val[type_val == 'red'], y_pred[type_val == 'red'], all_classes)
acc_white = calculate_per_class_accuracy(y_val[type_val == 'white'], y_pred[type_val == 'white'], all_classes)

x_indices = np.arange(len(all_classes))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x_indices - width / 2, acc_red, width, label='Wino Czerwone', color='#a52a2a', edgecolor='black')
ax.bar(x_indices + width / 2, acc_white, width, label='Wino Białe', color='#f4c430', edgecolor='black')
ax.set_ylabel('Skuteczność (%)')
ax.set_xlabel('Jakość')
ax.set_title('Skuteczność klasyfikacji per klasa')
ax.set_xticks(x_indices)
ax.set_xticklabels(all_classes)
ax.legend()
plt.tight_layout()
plt.savefig('skutecznosc_pelna.png', dpi=300)
plt.close()

plot_confusion_matrix(y_val, y_pred, all_classes, 'Macierz Pomyłek - Cały Zbiór', 'macierz_pomylek_calosc.png',
                      plt.cm.Greens)

mask_red = (type_val == 'red')
plot_confusion_matrix(y_val[mask_red], y_pred[mask_red], all_classes, 'Macierz Pomyłek - Wino Czerwone',
                      'macierz_pomylek_red.png', plt.cm.Reds)

mask_white = (type_val == 'white')
plot_confusion_matrix(y_val[mask_white], y_pred[mask_white], all_classes, 'Macierz Pomyłek - Wino Białe',
                      'macierz_pomylek_white.png', plt.cm.YlOrBr)