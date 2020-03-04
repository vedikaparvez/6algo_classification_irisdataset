# import dependencies
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

# load dataset and check contents
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# # shape
# print(dataset.shape)
# # head
# print(dataset.head(20))
# # descriptions
# print(dataset.describe())
# # class distribution
# print(dataset.groupby('class').size())

# visualise the data
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# create the sub models
estimators = []
model1 = LogisticRegression(solver='liblinear', multi_class='ovr')
estimators.append(('logistic', model1))
model2 = LinearDiscriminantAnalysis()
estimators.append(('linearda', model2))
model3 = KNeighborsClassifier()
estimators.append(('knclassifier', model3))
model4 = DecisionTreeClassifier()
estimators.append(('cart', model4))
model5 = GaussianNB()
estimators.append(('gaussian', model5))
model6 = SVC(gamma='auto')
estimators.append(('svm', model6))

# evaluate each model in turn
results = []
names = []
for name, model in estimators:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))