from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from .models import Parkinson

# Create your views here.
def index(request):
    return render(request,"index.html")

def signup(request):
    if request.method=="POST":
        first=request.POST['fname']
        last=request.POST['lname']
        user=request.POST['uname']
        e=request.POST['email']
        p1=request.POST['psw']
        p2=request.POST['psw-repeat']
        #Store the value in db
        u1=User.objects.create_user(first_name=first,last_name=last,
        email=e,username=user,password=p2)
        u1.save()
        return redirect('signin')
    else:
        return render(request,"signup.html")
    
def signin(request):
    if request.method=="POST":
        u=request.POST['uname']
        p=request.POST['psw']
        user=auth.authenticate(username=u,password=p)
        if user is not None:
            auth.login(request,user)
            return redirect('parkinson')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"signin.html")
    return render(request,"signin.html")

def test(request):
    fo=float(request.POST['MDVP:Fo(Hz)'])
    fhi=float(request.POST['MDVP:Fhi(Hz)'])
    flo=float(request.POST['MDVP:Flo(Hz)'])
    j=float(request.POST['MDVP:Jitter(%)'])
    jabs=float(request.POST['MDVP:Jitter(Abs)'])
    rap=float(request.POST['MDVP:RAP'])
    ppq=float(request.POST['MDVP:PPQ'])
    ddp=float(request.POST['Jitter:DDP'])
    s=float(request.POST['MDVP:Shimmer'])
    sdb=float(request.POST['MDVP:Shimmer(db)'])
    apq3=float(request.POST['Shimmer:APQ3'])
    apq5=float(request.POST['Shimmer:APQ5'])
    apq=float(request.POST['MDVP:APQ'])
    dda=float(request.POST['Shimmer:DDA'])
    nhr=float(request.POST['NHR'])
    hnr=float(request.POST['HNR'])
    rpde=float(request.POST['RPDE'])
    dfa=float(request.POST['DFA'])
    s1=float(request.POST['spread1'])
    s2=float(request.POST['spread2'])
    d2=float(request.POST['D2'])
    ppe=float(request.POST['PPE'])
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    parkinsons_data = pd.read_csv('static\datasets\parkinsons.csv')
    X = parkinsons_data.drop(columns=['name','status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)
    input_data = (fo,fhi,flo,j,jabs,rap,ppq,ddp,s,sdb,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,s1,s2,d2,ppe)

    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        result="The Person does not have Parkinsons Disease"
        z=Parkinson.objects.create(MDVP_Fo=fo,MDVP_Fhi=fhi,MDVP_Flo=flo,MDVP_Jitter=j,MDVP_Jitter_abs=jabs,MDVP_RAP=rap,MDVP_PPQ=ppq,Jitter_DDP=ddp,MDVP_Shimmer=s,MDVP_Shimmer_db=sdb,Shimmer_APQ3=apq3,Shimmer_APQ5=apq5,MDVP_APQ=apq,Shimmer_DDA=dda,NHR=nhr,HNR=hnr,status=False,RPDE=rpde,DFA=dfa,spread1=s1,spread2=s2,D2=d2,PPE=ppe)
        z.save()
       
    else:
        result="The Person has Parkinsons"
        z=Parkinson.objects.create(MDVP_Fo=fo,MDVP_Fhi=fhi,MDVP_Flo=flo,MDVP_Jitter=j,MDVP_Jitter_abs=jabs,MDVP_RAP=rap,MDVP_PPQ=ppq,Jitter_DDP=ddp,MDVP_Shimmer=s,MDVP_Shimmer_db=sdb,Shimmer_APQ3=apq3,Shimmer_APQ5=apq5,MDVP_APQ=apq,Shimmer_DDA=dda,NHR=nhr,HNR=hnr,status=True,RPDE=rpde,DFA=dfa,spread1=s1,spread2=s2,D2=d2,PPE=ppe)
        z.save()

    return render(request,"test.html",{'result':result})
    
def parkinson(request):
    return render(request,"parkinson.html")
    