from django.http import HttpResponse
from django.shortcuts import render
import joblib

#from django.utils.encoding import python_2_unicode_compatible

def home(request):
    return render(request,"home.html")


def results(request):
    dtree=joblib.load('dtree.sav')
    scaler=joblib.load('scaler.sav')
    labelencoder_context=joblib.load('labelencoder_context.sav')
    labelencoder_reasonstart=joblib.load('labelencoder_reasonstart.sav')
    labelencoder_reasonend=joblib.load('labelencoder_reasonend.sav')


    lis=[]
    lis.append(scaler.transform([[request.GET['Col2']]]))
    lis.append(scaler.transform([[request.GET['Col3']]]))
    lis.append(labelencoder_context.transform([request.GET['Col4']]))
    lis.append(request.GET['Col5'])
    lis.append(request.GET['Col6'])
    lis.append(scaler.transform([[request.GET['Col7']]]))
    lis.append(scaler.transform([[request.GET['Col8']]]))
    lis.append(scaler.transform([[request.GET['Col9']]]))
    lis.append(request.GET['Col10'])
    lis.append(labelencoder_reasonstart.transform([request.GET['Col11']]))
    lis.append(labelencoder_reasonend.transform([request.GET['Col12']]))
    date=request.GET['Col13']
    year,month,day=date.split('/')
    lis.append(scaler.transform([[year]]))
    lis.append(scaler.transform([[month]]))
    lis.append(scaler.transform([[day]]))
    print(lis)
    ans=dtree.predict([lis])
    if ans<0.5:
        res='Not Skipped'
    else:
        res='Skipped'

    return render(request,"result.html",{'res':res})
