import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

wine_data = load_wine()

X = wine_data.data
y = wine_data.target

# create a dataframe from the available wine data
df = pd.DataFrame(X, columns=wine_data.feature_names)
df['target'] = y
df['wine_class'] = df['target'].apply(lambda x: wine_data.target_names[x])

# filter the data and select only top 5 features
feature_indices = [3,5,6,10,11,12]
X_trained = X[:,feature_indices]

X_train, X_test, y_train, y_test = train_test_split(X_trained, y, test_size=0.2, random_state=42)

# using svm classifier, train the model and check the overall performance of the model
svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)

svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

performance = classification_report(y_test, y_pred)

print(f"Classification report: \n{performance}")