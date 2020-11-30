from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import pandas as pd

data = pd.read_csv("smartphoneList.txt",header=None)
df = pd.DataFrame(data)
dataTarget = df[561]
startTime = time.time()

X_train, X_test, y_train, y_test = train_test_split(df, dataTarget, test_size=0.3,train_size=0.7)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

endTime = time.time()-startTime

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(f"Total Time after: {endTime}")