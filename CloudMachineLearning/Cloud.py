

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import socket

root = tkinter.Tk()

root.title("Semantics of Data Mining Services in Cloud Computing")
root.geometry("1200x700")

global filename
global rf_acc
global X_train
global y_train
global dataset
global X_test
global y_test
global rf

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    
       

def preprocess():
    global X_train
    global y_train
    global dataset
    global X_test
    global y_test
    dataset = pd.read_csv(filename)
    y = dataset['Outcome']
    X = dataset.drop(['Outcome'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    text.delete('1.0', END)
    text.insert(END,"Dataset Length : "+str(len(dataset))+"\n")
    text.insert(END,"Random Forest Training on dataset length : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Random Forest Testing on dataset length : "+str(X_test.shape[0])+"\n")

def decisionTree():
    global rf
    global rf_acc
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test) 
    rf_acc = accuracy_score(y_test,y_pred)*100
    text.insert(END,"Random Forest Accuracy : "+str(rf_acc)+"\n")


def runServer():
    global rf
    if 'rf' not in globals():
        text.insert(END, "Error: Random Forest model is not trained. Please train the model first.\n")
        return
    headers = 'Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age'
    host = socket.gethostname()
    port = 5000
    server_socket = socket.socket()
    server_socket.bind((host, port))
    text.insert(END, "Server started. Waiting for client connections...\n")
    while True:   
        server_socket.listen(2)
        conn, address = server_socket.accept()
        data = conn.recv(1024).decode()
        f = open("test.txt", "w")
        f.write(headers+"\n"+str(data))
        f.close()
        text.insert(END,"from connected user: " + str(data)+"\n")
        test = pd.read_csv('test.txt')
        predict = rf.predict(test)
        data = str(predict[0])
        text.insert(END,"Disease Prediction " + str(data)+"\n")
        root.update_idletasks()
        conn.send(data.encode())


font = ('times', 18, 'bold')
title = Label(root, text='Semantics of Data Mining Services in Cloud Computing')
title.config(bg='wheat', fg='red')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')

upload = Button(root, text="Upload Diabetes Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(root)
pathlabel.config(bg='blue', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

preprocessButton = Button(root, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)  

treeButton = Button(root, text="Train Random Forest Algorithm", command=decisionTree)
treeButton.place(x=50,y=200)
treeButton.config(font=font1)


serverButton = Button(root, text="Start Cloud Server & Publish CCDM Service", command=runServer)
serverButton.place(x=50,y=250)
serverButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(root,height=28,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=150)
text.config(font=font1)

root.mainloop()
