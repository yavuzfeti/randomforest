# Pandas kütüphanesi pd kısaltmasıyla eklenerek import edildi.
import pandas as pd
# sklearn.model_selection kütüphanesinden train_test_split fonksiyonu eklenerek import edildi.
from sklearn.model_selection import train_test_split
# sklearn.ensemble kütüphanesinden RandomForestClassifier sınıfı eklenerek import edildi.
from sklearn.ensemble import RandomForestClassifier
# sklearn.metrics kütüphanesinden accuracy_score, confusion_matrix, classification_report fonksiyonları eklenerek import edildi.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# sklearn.metrics kütüphanesinden roc_curve, auc fonksiyonları eklenerek import edildi.
from sklearn.metrics import roc_curve, auc
# matplotlib.pyplot kütüphanesi plt kısaltmasıyla eklenerek import edildi.
import matplotlib.pyplot as plt
# sklearn.preprocessing kütüphanesinden label_binarize fonksiyonu eklenerek import edildi.
from sklearn.preprocessing import label_binarize
# sklearn.multiclass kütüphanesinden OneVsRestClassifier sınıfı eklenerek import edildi.
from sklearn.multiclass import OneVsRestClassifier

# 'iris.csv' dosyası okunarak iris_df adlı bir DataFrame'e atandı.
iris_df = pd.read_csv('iris.csv')

# Bağımsız değişkenler (X) ve bağımlı değişken (y) tanımlandı.
X = iris_df.drop('target', axis=1)
y = iris_df['target']

# Veri kümesi eğitim ve test setlerine bölündü. Test seti oranı %20, random_state=42 kullanılarak belirtildi.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier 100 ağaç ile oluşturuldu ve OneVsRestClassifier kullanılarak bir sınıflandırıcı oluşturuldu.
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# Eğitim verileri kullanılarak sınıflandırıcı eğitildi.
clf.fit(X_train, y_train)

# Test seti kullanılarak tahminler yapıldı.
y_pred = clf.predict(X_test)

# Test seti doğruluğu hesaplandı ve yazıldı.
accuracy = accuracy_score(y_test, y_pred)
print(f'Test seti doğruluk oranı: {accuracy * 100:.2f}%')

# Karmaşıklık matrisi hesaplandı ve yazıldı.
cm = confusion_matrix(y_test, y_pred)
print('Karmaşıklık Matrisi:')
print(cm)

# Sınıflandırma raporu hesaplandı ve yazıldı.
print('Sınıflandırma Raporu:')
print(classification_report(y_test, y_pred))

# Sınıflar için ROC eğrileri hesaplandı.
y_prob = clf.predict_proba(X_test)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC eğrileri çizildi.
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC eğrisi sınıf {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()

# Yeni bir çiçeğin sınıfı tahmin edildi.
new_flower = [[5.1, 3.5, 1.4, 0.2]]

predicted_class = clf.predict(new_flower)

# Tahmin edilen Iris türü yazıldı.
print(f"Tahmin edilen Iris türü: {predicted_class[0]}")