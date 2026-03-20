import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Titanic-Dataset (1).csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())
print(df[['Age','Cabin','Embarked']])
'''age is a numerical feature so use median because age have outliers'''
df['Age'] = df['Age'].fillna(df['Age'].median())

'''Cabin drop because so many null values'''
df.drop('Cabin',axis=1,inplace=True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print(df.isnull().sum())

'''EDA'''
'''Survival visuvalization'''
plt.figure(figsize=(6,4))

ax = sns.countplot(x = 'Survived',data = df, palette='Set2',hue='Survived')

plt.title("Survival count of titanic ship")
plt.xlabel("Survival status(0 = died,1 = survive)")
plt.ylabel("Number of people")

#add top of bars

for i in ax.patches:
    height = i.get_height()
    if height > 0:
        ax.annotate(f'{int(i.get_height())}',
                    (i.get_x() + i.get_width()/2.,i.get_height()),
                    ha='center',va='bottom')
plt.grid(axis='y',linestyle = '--',alpha=0.7)
plt.show()

'''gender vice survival rate'''
plt.figure(figsize=(6,4))
ax = sns.countplot(x = "Sex", palette='Set2',hue="Survived",data = df)

plt.title("Gender vice survival rate")
plt.xlabel("survival status (male and female) 0 = died, 1 = survive")
plt.ylabel("Number of people")

#change legend names
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,['Died','survived'],title="Status")

for i in ax.patches:
    height = i.get_height()
    if height > 0:
        ax.annotate(f'{int(i.get_height())}',
                    (i.get_x() + i.get_width()/2.,i.get_height()),
                    ha='center',va='bottom')
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.show()

'''survival rate for class wise'''
plt.figure(figsize=(6,4))

ax = sns.countplot(x='Pclass',data=df,palette='Set2',hue='Survived')
plt.title("survival rate for booking class")
plt.xlabel("classwise survival")
plt.ylabel("Number of people")

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,['Died','Survived'],title='status')
for i in ax.patches:
    height = i.get_height()
    if height > 0:
        ax.annotate(f'{int(i.get_height())}',
                    (i.get_x() + i.get_width()/2.,i.get_height()),
                    ha='center',va='bottom')
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.show()

'''age wise '''
plt.figure(figsize=(6,4))
sns.histplot(df['Age'],bins=30,kde=True,color='green',label='Age Histogram')
plt.title("age wise survival rate")
plt.xlabel("age")
plt.ylabel("Number of people")
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.legend(title="plot type")
plt.tight_layout()
plt.show()

'''Ticket fare wise survival'''
plt.figure(figsize=(6,4))
sns.histplot(df['Fare'],bins=30,kde=True,color='skyblue',label='ticket fare')
plt.title("Ticket Fare Distribution")
plt.xlabel("Ticket fare")
plt.ylabel("number of people")
plt.legend(title="plot type")
plt.tight_layout()
plt.show()

'''heatmap'''

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("heatmap")
plt.show()


'''Feature Engineering'''
#Family size - its very important for survival and add one is adding passenger also
df["Family size"] = df['SibSp'] + df['Parch'] + 1
print(df[["SibSp","Parch","Family size"]].head())

#2.Is alone
df['IsAlone'] = 0
df.loc[df["Family size"]==1,"IsAlone"] = 1
print(df[["Family size","IsAlone"]].head())

#Remove SibSb and Parch
df = df.drop(['SibSp','PassengerId','Parch','Name','Ticket'],axis=1)
print(df.columns)

#Convert Sex into str to numeric
print(df['Sex'].head())
df["Sex"] = df["Sex"].map({'male':0,"female":1})
print(df["Sex"].unique())

print('________________________')

#Convert Embarked into Cat to num
df = pd.get_dummies(df, columns=['Embarked'],drop_first=True)
# print(df['Embarked'].dtype)
print(df.columns)
print(df.info())

#Split into groups
df["Age Group"] = pd.cut(df['Age'],bins=[0,12,20,40,60,80],labels=[0,1,2,3,4])
df["FareGroup"] = pd.qcut(df['Fare'],4,labels=[0,1,2,3])

print(df.isnull().sum())

df["Age Group"] = df["Age Group"].astype(int)
df["FareGroup"] = df["FareGroup"].astype(int)

print(df.info())

'''model'''
X = df.drop("Survived",axis=1)
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

'''Logistic Regression'''
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
log_result = logreg.predict(X_test)

'''Evaluate Logistic Regression'''
print(f'Logistic Regression Accuracy:{accuracy_score(y_test,log_result)}')
print(f'\nconfusion matrix\n{confusion_matrix(y_test,log_result)}')
print(f'\nClassification Report\n{classification_report(y_test,log_result)}')

'''Random Forest Classifier '''
ranforclass = RandomForestClassifier(n_estimators=100,random_state=42)
ranforclass.fit(X_train,y_train)
y_rfc = ranforclass.predict(X_test)

print("Random Forest Classifier accuracy:",accuracy_score(y_test,y_rfc))
print("Confuison Matrix:",confusion_matrix(y_test,y_rfc))
print("Classification Report:",classification_report(y_test,y_rfc))

'''ROC - AUC Score'''
roc_score_lg = roc_auc_score(y_test,log_result)
roc_score_rfc = roc_auc_score(y_test,y_rfc)

print("roc_auc_log_reg",roc_score_lg)
print("roc_auc_random_forest_classification:",roc_score_rfc)
