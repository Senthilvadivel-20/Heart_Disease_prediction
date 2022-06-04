import pandas as pd #Dataframe manipulation Library
import numpy as np #Data manipulation Library

#Scikit Learn libraries for Logisitic Regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("./csv/heart_2020_cleaned.csv")


HeartDiseaseEncoder = LabelEncoder()
SmokingEncoder = LabelEncoder()
AlcoholDrinkingEncoder = LabelEncoder()
StrokeEncoder = LabelEncoder()
DiffWalkingEncoder = LabelEncoder()
SexEncoder = LabelEncoder()
AgeCategoryEncoder = LabelEncoder()
RaceEncoder = LabelEncoder()
DiabeticEncoder = LabelEncoder()
PhysicalActivityEncoder = LabelEncoder()
GenHealthEncoder = LabelEncoder()
AsthmaEncoder = LabelEncoder()
KidneyDiseaseEncoder = LabelEncoder()
SkinCancerEncoder = LabelEncoder()

df['HeartDisease'] = HeartDiseaseEncoder.fit_transform(df['HeartDisease'])
df['Smoking'] = SmokingEncoder.fit_transform(df['Smoking'])
df['AlcoholDrinking'] = AlcoholDrinkingEncoder.fit_transform(df['AlcoholDrinking'])
df['Stroke'] = StrokeEncoder.fit_transform(df['Stroke'])
df['DiffWalking'] = DiffWalkingEncoder.fit_transform(df['DiffWalking'])
df['Sex'] = SexEncoder.fit_transform(df['Sex'])
df['AgeCategory'] = AgeCategoryEncoder.fit_transform(df['AgeCategory'])
df['Race'] = RaceEncoder.fit_transform(df['Race'])
df['Diabetic'] = DiabeticEncoder.fit_transform(df['Diabetic'])
df['PhysicalActivity'] = PhysicalActivityEncoder.fit_transform(df['PhysicalActivity'])
df['GenHealth'] = GenHealthEncoder.fit_transform(df['GenHealth'])
df['Asthma'] = AsthmaEncoder.fit_transform(df['Asthma'])
df['KidneyDisease'] = KidneyDiseaseEncoder.fit_transform(df['KidneyDisease'])
df['SkinCancer'] = SkinCancerEncoder.fit_transform(df['SkinCancer'])


x = df[['BMI', 'AlcoholDrinking', 'Stroke', 'Sex', 'AgeCategory', 'Diabetic', 'SleepTime', 'Asthma',
       'KidneyDisease']]

y = df['HeartDisease']

#Splitting the dataset into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size= 0.3,
                                                    random_state=41)

def log():
    algo = "Logistic Regression"
    model = LogisticRegression() #Initializing the Logistic Regression Model
    model.fit(x_train,y_train)  #Training the Logistic Regression Model
    y_pred = model.predict(x_test) #Predicting the results for test dataset
    accuracy = accuracy_score(y_test, y_pred) #Calculating the accuracy score of the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return [accuracy*100, precision, recall, f1, algo]

def random():
    from sklearn.ensemble import RandomForestClassifier
    # create regressor object
    algo = "Random Forest Classifier"
    regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
    # fit the regressor with x and y data
    regressor.fit(x_train, y_train) 
    y_pred = regressor.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) #Calculating the accuracy score of the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return [accuracy*100, precision, recall, f1, algo]

def knn():
    algo = "K-NeighborsClassifier" 
    # Import necessary modules
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=7)
    
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) #Calculating the accuracy score of the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return [accuracy*100, precision, recall, f1, algo]

def nav():
    algo = "Naive Bayes"
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) #Calculating the accuracy score of the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return [accuracy*100, precision, recall, f1, algo]

def dec():
    from sklearn.tree import DecisionTreeClassifier
    clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
    # Performing training
    algo = "Decision Tree Classifier"
    clf_gini.fit(x_train, y_train)
    y_pred = clf_gini.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) #Calculating the accuracy score of the model
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test, y_pred,average='micro')
    return [accuracy*100, precision, recall, f1, algo]

def accuracy():
    acc = [log(), random() ,knn() , nav(), dec()]
    return acc

# print(accuracy())