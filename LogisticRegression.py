from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Tk,Label, Button,Entry, messagebox
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from PIL import Image,ImageTk
import matplotlib.pyplot as plt

#Reading the Data
DataPath = ('Loan_Status.csv')
data = pd.read_csv(DataPath)

# # Checking for Missing Values

# # (insa)check if there are any missing values in the dataframe
# # (any)check if there are any missing values in any of the columns
# # Count the number of missing values in each column
# # print(data.isna().any())

# # # Data Exploration
# # # get the number of unique values in each column
# # print(data.nunique())

# # # display the first and last few rows of the dataframe
# # data.head()
# # data.tail()

# # # (Shape)get the number of rows and columns in the dataframe
# # print("Number of Rows",data.shape[0])
# # print("Number of Columns",data.shape[1])

# # Data Cleaning

# data.info()

# # function is used to count the number of missing values in each column
# data.isnull().sum()
# # calculates the percentage of null values in each column of the dataset.
# print(data.isnull().sum()*100 / len(data))

# # Drop any rows in the dataframe that contain missing values.
data=data.dropna()
# print(data.isnull().sum())

# # Display descriptive statistics about the dataframe such as the mean, standard deviation.
# print(data.describe())

# Adjustment to numerical values by replacing  'N' with 0 and 'Y' with 1
data.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)

# # counts the unique values 
# print(data['Dependents'].value_counts())

# # replaces any '3+' in the 'Dependents' column with the value 4
data['Dependents'].replace('3+',4,inplace=True)

data.isnull().sum()*100 / len(data)

# print(data['Self_Employed'].mode()[0])

# 'inplace=True' parameter ensures that the changes are made directly to the 'data' dataframe instead of creating a new one
data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

data=data.drop('Loan_ID',axis=1)
# print(data.head(1))
x = data[['CoapplicantIncome', 'LoanAmount']].values
x = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

# print(x.shape)
# print(y.shape)

# contains a list of column names that need to be standardized
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

#  StandardScaler() function used to standardize the numerical variables
#  reducing the number of iterations required to reach a solution.
st = StandardScaler()

# (fit_transform)function standardizes the columns listed in the 'cols'
x[cols]=st.fit_transform(x[cols])
# print(x)

#  initializes an empty dictionary 'model_df' to store model results.
model_df={}
#train set and test set
def model_val(model,x,y):
    # function splits the data into training and testing sets
    # train_test_split function from the sklearn library to split
    # the data into training and testing sets, 
    # with 20% of the data used for testing and the rest used for training.
    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                   test_size=0.20,
                                                   random_state=42)
    
    #  initializes an empty dictionary 'model_df' to store model results.
    model.fit(x_train,y_train)
    
    # function creates predictions on the test set.
    y_pred=model.predict(x_test)
    # print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")

    #  function outputs the accuracy score of the model as a string
    return str (accuracy_score(y_test,y_pred)*100)
    
    
def Accuracy():
    model = LogisticRegression()
    result= model_val(model,x,y)
    acc.set(result)
    

# Random Forest model with specific parameters such as the number of estimators,
# minimum samples per split, minimum samples per leaf, maximum features, and maximum depth.
# It can handle missing values and noisy data.
# It is less prone to overfitting than other classification algorithms.
rf = RandomForestClassifier(
 n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5
)

# Train the 'rf' model on 'x' and 'y'.
rf.fit(x,y)

# save the model to a file called 'loan_status_predict'
joblib.dump(rf,'loan_status_predict')

model = joblib.load('loan_status_predict')


master =Tk()
master.title("Loan Status Prediction")
newGeometry = '1000x600+250+100'
master.geometry(newGeometry)
master.configure(bg='#fff')
label = Label(master,text = "Loan Status Prediction", fg="#57a1f8",bg='white' ,font=('sans-serif',30,'bold')
              ).place(x=240,y=5)
 
path ="loanimg.png"
img = ImageTk.PhotoImage(Image.open(path))
photo =Label(master,image=img,border=0,bg='white').place(x=1,y=90)

def delete_all():
    for widget in master.winfo_children():
        if isinstance(widget, (tk.Entry)):
            widget.delete(0, tk.END)
        

gender = IntVar()
married = IntVar()
dependents = Entry(master)
education = IntVar()
self_employee = IntVar()
app_income = Entry(master)
coapp_income = Entry(master)
loan_amount = Entry(master)
loan_amount_term = Entry(master)
credit_history = Entry(master)
prop_area = IntVar()


dependents.place(x=680,y=145)
app_income.place(x=680,y=240)
coapp_income.place(x=680,y=280)
loan_amount.place(x=680,y=320)
loan_amount_term.place(x=680,y=360)
credit_history.place(x=680,y=400)


def show_entry():
    Gender = gender.get()
    Married = married.get()
    Dependents = dependents.get()
    Education = education.get()
    Self_Employee = self_employee.get()
    App_Income = app_income.get()
    CoApp_Income = coapp_income.get()
    Loan_Amount = loan_amount.get()
    Loan_Amount_Term = loan_amount_term.get()
    Credit_History = credit_history.get()
    Prop_Area = prop_area.get()
    
    
    if( Dependents == "" ):
        messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "Choose DEPENDANTS !!!")
        dependents.focus_set() 
    elif(App_Income == ""):
        messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "APPLICANT INCOME can't empty !!!")
        app_income.focus_set()
    elif(CoApp_Income == ""):
         messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "CO-APPLICANT INCOME can't empty!!!")
         coapp_income.focus_set()
    elif (Loan_Amount.isdigit() == False):
        messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "Enter LOAN AMOUNT in numbers only !!!")
        loan_amount.focus_set()
    elif (Loan_Amount_Term== ""):
        messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "LOAN TERM can't empty!!!")
        loan_amount_term.focus_set()
    elif(Credit_History== ""):
        messagebox.showinfo("MACHINE LEARNING LOAN PREDICTOR.", "CREDIT HISTORY can't empty!!!")
        credit_history.focus_set()



    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
    'Gender':Gender,
    'Married':Married,
    'Dependents':Dependents,
    'Education':Education,
    'Self_Employed':Self_Employee,
    'ApplicantIncome':App_Income,
    'CoapplicantIncome':CoApp_Income,
    'LoanAmount':Loan_Amount,
    'Loan_Amount_Term':Loan_Amount_Term,
    'Credit_History':Credit_History,
    'Property_Area':Prop_Area
},index=[0])
    result = model.predict(df)

    if result == 1:
        Label(master, text="Loan Approved", fg='red' ,bg='white',font=('Helvetica',15,'bold')).place(x=320,y=480)
    else:
        Label(master, text="Loan Not Approved" ,fg='red' ,bg='white',font=('Helvetica',15,'bold')).place(x=320,y=480)
    

radio_style = ttk.Style()
myColor = 'white'
radio_style.configure('Wild.TRadiobutton', background=myColor, foreground='#57a1f8', font='Helvetica 13 bold')    

Label(master,text = "Gender" ,fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=75)
ttk.Radiobutton(master, text = "Male", variable = gender, value = 1,style='Wild.TRadiobutton').place(x=600,y=75)
ttk.Radiobutton(master, text = "Female", variable = gender, value = 0,style='Wild.TRadiobutton').place(x=690,y=75)

Label(master,text = "Married",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=105)
ttk.Radiobutton(master, text = "Married", variable = married, value = 1,style='Wild.TRadiobutton').place(x=600,y=105)
ttk.Radiobutton(master, text = "Single", variable = married, value = 0,style='Wild.TRadiobutton').place(x=690,y=105)

Label(master,text = "Dependents [0...4]",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=145)

Label(master,text = "Education" ,fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=180)
ttk.Radiobutton(master, text = "Graduate", variable = education, value = 1,style='Wild.TRadiobutton').place(x=600,y=180)
ttk.Radiobutton(master, text = "Not Graduate", variable = education, value = 0,style='Wild.TRadiobutton').place(x=720,y=180)

Label(master,text = "Self_Employed",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=210)
ttk.Radiobutton(master, text = "Yes", variable = self_employee, value = 1,style='Wild.TRadiobutton').place(x=650,y=210)
ttk.Radiobutton(master, text = "No", variable = self_employee, value = 0,style='Wild.TRadiobutton').place(x=750,y=210)

Label(master,text = "ApplicantIncome",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=250)

Label(master,text = "CoapplicantIncome",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=290)

Label(master,text = "LoanAmount",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=330)

Label(master,text = "Loan_Amount_Term",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=370)

Label(master,text = "Credit_History",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=410)

Label(master,text = "Property_Area",fg='#57a1f8' ,bg='white',font=('Helvetica',11,'bold')).place(x=353,y=440)
ttk.Radiobutton(master, text = "Rural", variable = prop_area, value = 0,style='Wild.TRadiobutton').place(x=600,y=440)
ttk.Radiobutton(master, text = "Semiurban", variable = prop_area, value = 1,style='Wild.TRadiobutton').place(x=690,y=440)
ttk.Radiobutton(master, text = "Urban", variable = prop_area, value = 2,style='Wild.TRadiobutton').place(x=820,y=440)

def showdatagram():
 xaxis = data[['ApplicantIncome', 'LoanAmount']].values
 yaxis = data['Loan_Status'].values
 plt.figure(figsize=(8, 6))
 plt.scatter(xaxis[yaxis == 0][:, 0], xaxis[yaxis == 0][:, 1], color='b', label='Yes')
 plt.scatter(xaxis[yaxis == 1][:, 0], xaxis[yaxis == 1][:, 1], color='r', label='No')
 plt.xlabel('Applicant Income')
 plt.ylabel('Loan Amount')
 plt.legend()

 plt.show()


Button(master,text="Predict" , width=30,pady=7,bg='#57a1f8',fg='white',border=0, font=('Helvetica',11,'bold'),command=show_entry).place(x=30,y=480)
Button(master,text="Show Datagram" , width=30,pady=7,bg='#57a1f8',fg='white',border=0 ,font=('Helvetica',11,'bold'),command=showdatagram).place(x=30,y=530)
button3=Button(master,text="Accuracy" , width=30,pady=7,bg='#57a1f8',fg='white',border=0 ,font=('Helvetica',11,'bold'),command=Accuracy).place(x=550,y=480)
delete_button = Button(master, text="CLEAR", width=30,pady=7,bg='#57a1f8',fg='white',border=0 ,font=('Helvetica',11,'bold'),command=delete_all).place(x=550,y=530)



acc=StringVar()
myoutput=tk.Label(master,textvariable=acc,fg='red' ,bg='white',font=('Helvetica',11,'bold')).place(x=845,y=485)

mainloop()
