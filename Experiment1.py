import pandas as pd
import pydotplus
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Social_Network_Ads.csv')
print(data)

feature_cols = ['Age', 'EstimatedSalary']
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.write_png('decisiontree.png'))
