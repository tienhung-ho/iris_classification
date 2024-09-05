from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import joblib

iris = datasets.load_iris()
X = iris.data
X_columns = iris.feature_names
y = iris.target

X = pd.DataFrame(X, columns=X_columns)
print(X.describe())
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" % accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
index=['setosa', 'versicolor', 'virginica'],
columns=['setosa', 'versicolor', 'virginica'])

plt.figure(figsize=(5.5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('GaussianNB \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

joblib.dump(model, "native_bayes_model.pkl")