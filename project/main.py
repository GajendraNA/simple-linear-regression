import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 

def welcome():
    print("Welcome to the Data Analysis Project!")
    print("Press enter key to proceed.")
    input()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file found in the directory'
    return csv_files

def display_and_select_csv(csv_files):
    i = 0
    for file_name in csv_files:
        print(i , '   ', file_name)
        i+=1
    return csv_files[int(input("Select file to create ML model"))]


def graph(X_train,Y_train, regressionObject, X_test, Y_test, Y_pred):
    plt.scatter(X_train,Y_train,color = 'red', label= 'training data')
    plt.plot(X_train,regressionObject.predict(X_train), color = 'blue', label= 'Best Fit')
    plt.scatter(X_test,Y_test,color = 'green', label= 'testing data')
    plt.scatter(X_test,Y_pred,color = 'black', label= 'predicted test data')
    plt.title("Salary vs experience")
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()

def main():
    welcome()
    try:
        csv_files = checkcsv()
        if csv_files=='No csv file found in the directory':
            raise FileNotFoundError('Np csv file in the directory')
        csv_file=display_and_select_csv(csv_files)
        print(csv_file,"is selected")
        print("reading csv file")
        print("Creating Dataset")
        dataset=pd.read_csv(csv_file)
        print("Dataset created successfully")
        X=dataset.iloc[:,:-2].values
        Y=dataset.iloc[:,-1].values
        s= float(input("Enter test size in decimal (0-1)"))
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
        print("Model creation is in progress")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model created successfully")
        print("Press Enter to predict test data in trained model")
        input()
        
        Y_pred=regressionObject.predict(X_test)
        i=0
        
        while i< len(X_test):
            print("For input ",X_test[i]," the predicted output is ",Y_pred[i]," and actual output is ",Y_test[i])
            i+=1
        print("Press Enter key to see above result in graphical format")
        input()
        graph(X_train,Y_train, regressionObject, X_test, Y_test, Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("Our model is %2.2f%%  accurate" % (r2* 100))
        print("Now you can predict the salary of an employee using the model")
        print("Enter the years of experience of the employee, separated by commas")

        exp= [float(e) for e in input().split(',')]
        ex=[]
        for x in exp:
            ex.append([x])
        experience = np.array(ex)
        salaries= regressionObject.predict(experience)
        plt.scatter(experience,salaries, color='black')
        plt.xlabel("Years of experience")
        plt.ylabel("Salaries")
        plt.legend()
        plt.show()

        d=pd.DataFrame({'Experience' :exp, 'Salaries': salaries})
        print(d)

    except FileNotFoundError:
        print('No csv file found in the directory. Please add a csv file and try again.')
        print('Press Enter to exit.')
        input()
        exit()


if __name__ == "__main__":
    main()
    input()