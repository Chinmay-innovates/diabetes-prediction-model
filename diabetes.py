import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading diabetes dataset to pandas dataframe
diabetes_dataset = pd.read_csv("./diabetes.csv")
print(diabetes_dataset.head(5))
print(f"no of rows: {diabetes_dataset.shape}")
#statistical measure of data
print(diabetes_dataset.describe())

print(f"count of 0 and 1 {diabetes_dataset['Outcome'].value_counts()}")
# 0 -> non diabetic
# 1 -> diabetic

print(diabetes_dataset.groupby('Outcome').mean())

#separating data and labels
X =  diabetes_dataset.drop(columns='Outcome',axis=1)
Y =  diabetes_dataset['Outcome']

print("dropping data and labels")
print(f" X: {X}")
print(f" Y: {Y}")

#data standardization
scaler = StandardScaler()

standardized_dataset = scaler.fit_transform(X)

print(f'standardized_dataset: {standardized_dataset}')

X = standardized_dataset
Y = diabetes_dataset['Outcome']
print(f'std_X: {X}, the same Y: {Y}')

#split into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) 
print(X.shape,X_train.shape,X_test.shape)

#training the modal
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train,Y_train)

#model evaluation
#accracy score of training data
X_train_pred = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred,Y_train)
print('accracy score of training data :',training_data_accuracy * 100)

#accracy score of test data
X_test_pred = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred,Y_test)
print('accracy score of test data :',test_data_accuracy * 100)


#>>>>>>>>>>>>>>>>
#>> USER INPUT >>
#>>>>>>>>>>>>>>>>

# making the predective system
input_data_0 = (4,110,92,0,0,37.6,0.191,30) # non-diabetic
input_data_1 = (
    6,     #Pregnancies
    148,   #Glucose
    72,    #BloodPressure
    35,    #SkinThickness
    0,     #Insulin
    33.6,  #BMI
    0.627, #DiabetesPedigreeFunction
    50     #Age
    ) #diabetic

#change input to numpy array
input_data_as_numpy_array =np.asarray(input_data_1)

#reshape the array as predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardized the input data
std_data = scaler.transform(input_data_reshaped)
print(f"standardized input data: {std_data}")

prediction = classifier.predict(std_data)

print(f"prediction: {prediction}")

print("The person is non-diabetic.") if (prediction[0] == 0) else print("The person is diabetic.")

#>>>OUTPUT>>>

#  Pregnancies  Glucose  BloodPressure  SkinThickness  ...   BMI DiabetesPedigreeFunction  Age  Outcome
# 0            6      148             72             35  ...  33.6                     0.627   50        1
# 1            1       85             66             29  ...  26.6                     0.351   31        0
# 2            8      183             64              0  ...  23.3                     0.672   32        1
# 3            1       89             66             23  ...  28.1                     0.167   21        0
# 4            0      137             40             35  ...  43.1                     2.288   33        1

# [5 rows x 9 columns]
# no of rows: (768, 9)
#        Pregnancies     Glucose  BloodPressure  ...  DiabetesPedigreeFunction         Age     Outcome
# count   768.000000  768.000000     768.000000  ...                768.000000  768.000000  768.000000 
# mean      3.845052  120.894531      69.105469  ...                  0.471876   33.240885    0.348958 
# std       3.369578   31.972618      19.355807  ...                  0.331329   11.760232    0.476951 
# min       0.000000    0.000000       0.000000  ...                  0.078000   21.000000    0.000000 
# 25%       1.000000   99.000000      62.000000  ...                  0.243750   24.000000    0.000000 
# 50%       3.000000  117.000000      72.000000  ...                  0.372500   29.000000    0.000000 
# 75%       6.000000  140.250000      80.000000  ...                  0.626250   41.000000    1.000000 
# max      17.000000  199.000000     122.000000  ...                  2.420000   81.000000    1.000000 

# [8 rows x 9 columns]
# count of 0 and 1 Outcome
# 0    500
# 1    268

# Name: count, dtype: int64
#          Pregnancies     Glucose  BloodPressure  ...        BMI  DiabetesPedigreeFunction        Age
# Outcome                                          ...
# 0           3.298000  109.980000      68.184000  ...  30.304200                  0.429734  31.190000 
# 1           4.865672  141.257463      70.824627  ...  35.142537                  0.550500  37.067164 

# [2 rows x 8 columns]
# dropping data and labels
#  X: 
# Pregnancies  Glucose  BloodPressure  ...   BMI  DiabetesPedigreeFunction  Age
# 0              6      148             72  ...  33.6                     0.627   50
# 1              1       85             66  ...  26.6                     0.351   31
# 2              8      183             64  ...  23.3                     0.672   32
# 3              1       89             66  ...  28.1                     0.167   21
# 4              0      137             40  ...  43.1                     2.288   33
# ..           ...      ...            ...  ...   ...                       ...  ...
# 763           10      101             76  ...  32.9                     0.171   63
# 764            2      122             70  ...  36.8                     0.340   27
# 765            5      121             72  ...  26.2                     0.245   30
# 766            1      126             60  ...  30.1                     0.349   47
# 767            1       93             70  ...  30.4                     0.315   23

# [768 rows x 8 columns]
#  Y: 
# 0      1
# 1      0
# 2      1
# 3      0
# 4      1
#       ..
# 763    0
# 764    0
# 765    0
# 766    1
# 767    0
# Name: Outcome, Length: 768, dtype: int64

# standardized_dataset: [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
#    1.4259954 ]
#  [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
#   -0.19067191]
#  [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
#   -0.10558415]
#  ...
#  [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
#   -0.27575966]
#  [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
#    1.17073215]
#  [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
#   -0.87137393]]

# std_X: [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
#    1.4259954 ]
#  [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
#   -0.19067191]
#  [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
#   -0.10558415]
#  ...
#  [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
#   -0.27575966]
#  [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
#    1.17073215]
#  [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
#   -0.87137393]],
# the same Y: 
# 0      1
# 1      0
# 2      1
# 3      0
# 4      1
#       ..
# 763    0
# 764    0
# 765    0
# 766    1
# 767    0
# Name: Outcome, Length: 768, dtype: int64
# (768, 8) (614, 8) (154, 8)

# accracy score of training data : 78.66449511400651
# accracy score of test data : 77.27272727272727

# standardized input data: [[ 0.63994726  0.84832379  0.14964075  0.90726993 -0.69289057  0.20401277   
#    0.46849198  1.4259954 ]]
# prediction: [1]
# The person is diabetic.