from django.shortcuts import render
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from heart import acc
# from heart.heart.acc import accuracy

df = pd.read_csv("./csv/heart_2020_cleaned.csv")

HeartDiseaseEncoder = LabelEncoder()

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

df['HeartDisease'] = HeartDiseaseEncoder.fit_transform(df['HeartDisease'])

def home(request):
    return render(request,'Home.html')

def result(request):


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

    global BMI, AlcoholDrinking, Stroke, Sex, AgeCategory, Diabetic, SleepTime, Asthma,KidneyDisease, inputs

    BMI = request.GET['BMI']
    AlcoholDrinking = request.GET['AlcoholDrinking']
    Stroke = request.GET['Stroke']
    Sex = request.GET['Sex']
    AgeCategory = request.GET['AgeCategory']
    Diabetic = request.GET['Diabetic']
    SleepTime = request.GET['SleepTime']
    Asthma = request.GET['Asthma']
    KidneyDisease = request.GET['KidneyDisease']

    AlcoholDrinking = AlcoholDrinkingEncoder.transform([AlcoholDrinking])[0]
    Stroke = StrokeEncoder.transform([Stroke])[0]
    Sex = SexEncoder.transform([Sex])[0]
    AgeCategory = AgeCategoryEncoder.transform([AgeCategory])[0]
    Diabetic = DiabeticEncoder.transform([Diabetic])[0]
    Asthma = AsthmaEncoder.transform([Asthma])[0]
    KidneyDisease = KidneyDiseaseEncoder.transform([KidneyDisease])[0]    



    inputs = [BMI, AlcoholDrinking, Stroke, Sex, AgeCategory, Diabetic, SleepTime, Asthma,KidneyDisease]


    model = joblib.load('F:\Major Project\Heart\heart\ML\\logistics.sav')

    ans = model.predict([inputs])

    ans = HeartDiseaseEncoder.inverse_transform([ans])[0]

    log_score = acc.log()
    accuracy = log_score[0]
    precision = log_score[1]
    recall = log_score[2]
    f1_score = log_score[3]
    algo = log_score[4]



    return render(request,'result.html',locals())


def acc_chart(request):

    # acc_label = ['Logistics Regression', 'Random Forest','KNN', 'Naive Bayes',  'Decision Tree']
    # acc_value = acc.accuracy()

    return render(request,'acc_chart.html',locals())

def random_forest(request):
    model = joblib.load('F:\Major Project\Heart\heart\ML\RandomForestClassifier.sav')

    ans = model.predict([inputs])

    ans = HeartDiseaseEncoder.inverse_transform([ans])[0]

    ran_score = acc.random()
    accuracy = ran_score[0]
    precision = ran_score[1]
    recall = ran_score[2]
    f1_score = ran_score[3]
    algo = ran_score[4]

    return render(request,'result.html',locals())

def knn_algo(request):
    model = joblib.load('F:\Major Project\Heart\heart\ML\knn_cls.sav')
    ans = model.predict([inputs])
    ans = HeartDiseaseEncoder.inverse_transform([ans])[0]

    knn_score = acc.knn()
    accuracy = knn_score[0]
    precision = knn_score[1]
    recall = knn_score[2]
    f1_score = knn_score[3]
    algo =  knn_score[4]

    return render(request,'result.html',locals())

def naive_bayes(request):
    model = joblib.load('F:\Major Project\Heart\heart\ML\\nav_bys.sav')
    ans = model.predict([inputs])
    ans = HeartDiseaseEncoder.inverse_transform([ans])[0]
    naive_score = acc.nav()
    accuracy = naive_score[0]
    precision = naive_score[1]
    recall = naive_score[2]
    f1_score = naive_score[3]
    algo = naive_score[4]
    return render(request,'result.html',locals())

def decision_tree(request):
    model = joblib.load('F:\Major Project\Heart\heart\ML\\decision_tree.sav')
    ans = model.predict([inputs])
    ans = HeartDiseaseEncoder.inverse_transform([ans])[0]
    decision_score = acc.dec()
    accuracy = decision_score[0]
    precision = decision_score[1]
    recall = decision_score[2]
    f1_score = decision_score[3]
    algo = decision_score[4]
    return render(request,'result.html',locals())



def comparision_table(request):
    value = acc.accuracy()
    return render(request,'comparision_table.html',locals())

    



