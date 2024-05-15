#loan_id: Kredi ID - Her krediyi benzersiz bir şekilde tanımlayan bir numara.
#no_of_dependents: Bağımlıların Sayısı - Kredi alıcısının mali olarak desteklemek zorunda olduğu kişi sayısı.
#education: Eğitim - Kredi alıcısının eğitim seviyesi.
#self_employed: Kendi İşinde Çalışma Durumu - Kredi alıcısının kendi işinde çalışıp çalışmadığı.
#income_annum: Yıllık Gelir - Kredi alıcısının yıllık geliri.
#loan_amount: Kredi Miktarı - Alınan kredi miktarı.
#loan_term: Kredi Süresi - Kredinin geri ödeme süresi.
#cibil_score: CIBIL Skoru - Kredi alıcısının kredi geçmişini değerlendiren bir puan.
#CIBIL Skoru, bir bireyin kredi geçmişi ve geri ödeme davranışı temel alınarak hesaplanan üç haneli bir sayıdır1. Skor, 300 ile 900 arasında değişir1. Skorun yüksek olması, daha iyi bir kredi profili olduğunu gösterir1.
#CIBIL Skoru’nun yorumlanması genellikle aşağıdaki gibi yapılır:
#300-549: Düşük riskli olarak kabul edilir. Bu skora sahip kişiler genellikle kredi başvurularında reddedilir.
#550-649: Orta riskli olarak kabul edilir. Bu skora sahip kişilerin kredi başvuruları bazen kabul edilir, ancak genellikle daha yüksek faiz oranlarına tabidir.
#650-749: Düşük riskli olarak kabul edilir. Bu skora sahip kişiler genellikle kredi başvurularında kabul edilir ve makul faiz oranlarına tabidir.
#750-900: Çok düşük riskli olarak kabul edilir. Bu skora sahip kişilerin kredi başvuruları genellikle kabul edilir ve en düşük faiz oranlarına tabidir.


#residential_assets_value: Konut Varlıklarının Değeri - Kredi alıcısının sahip olduğu konut varlıklarının değeri.
#commercial_assets_value: Ticari Varlıkların Değeri - Kredi alıcısının sahip olduğu ticari varlıkların değeri.
#luxury_assets_value: Lüks Varlıkların Değeri - Kredi alıcısının sahip olduğu lüks varlıkların değeri.
#bank_asset_value: Banka Varlıklarının Değeri - Kredi alıcısının bankadaki varlıklarının değeri.
#loan_status: Kredi Durumu - Kredinin mevcut durumu (örneğin, ödenmiş, ödenmemiş, gecikmiş vb.).

#============================================================

# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv('miull ödev/loan_approval_dataset-2.csv')
df.head()

#============================================================



def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum()) #eksik deger var mı? varsa kac tane?
    #print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05,0.25, 0.50,0.75, 0.95, 0.99, 1])) # sayısal değişkenlerin ceyrekliklerinin incelenmes

# ============================================================


check_df(df)

#eksik değer var mı
df.head()
df.describe().T

df.isnull().sum()
# eksik gozlem var mı yok mu sorgusu 2.sorgu
df.isnull().values.any()

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=11, car_th=19):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"] # 0,1,2
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "0"] # name
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # cat_cols tüm object  veri tipini tuttugu için içerisinde cat_but_car bulunabilir.

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat] #numerik_gorunumlu kategorikler hariç

    print(f"Observations: {dataframe.shape[0]}") # satır
    print(f"Variables: {dataframe.shape[1]}") # sutun
    print(f'cat_cols: {len(cat_cols)}') # categorik degişken sayısı
    print(f'num_cols: {len(num_cols)}') # numerik değişkenler
    print(f'cat_but_car: {len(cat_but_car)}') # categorik fakat kardinal
    print(f'num_but_cat: {len(num_but_cat)}') # numerik görünümlü kategorik

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

##################eksik değerleri KNN ile doldurmak
##################################################
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) #false true şeklinde veriyor cıktıyı 1 ile çarparak çarparak değiştiridim
dff =pd.get_dummies(df[cat_cols + num_cols], drop_first=True)*1

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) #0 ve 1 den cıkarttık
# hangi değerlerimiz eksikti onu gördüm .
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df["no_of_dependents"] = dff[["no_of_dependents"]]
#df.loc[df[" no_of_dependents"].isnull(), [" no_of_dependents", " no_of_dependents"]]
#df.loc[df[" no_of_dependents"].isnull()] # bu kodlar hangi değişkene ne atadı diye bakıyormuş.
df["income_annum"] = dff[["income_annum"]]
df["luxury_assets_value"] = dff[["luxury_assets_value"]]
df["residential_assets_value"] = dff[["residential_assets_value"]]
df["commercial_assets_value"] = dff[["commercial_assets_value"]]
missing_values_table(df) #artık eksik gözlem yok
df.head()



#aykırı gözlemler neler ve onları silmeden sınırlandırmak
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

##Yeni değişkenler oluşturma
df.head()

df.sort_values("cibil_score", ascending=True).head()
df.sort_values("cibil_score", ascending=False).head()
df.head(220)


df.loc[(df['cibil_score'] >= 0) & (df['cibil_score'] < 500), 'New_cibil_score'] = 'Low'
df.loc[(df['cibil_score'] >= 500), 'New_cibil_score'] = 'high'

df.sort_values("loan_term",ascending=False).head()
df.loc[(df['loan_term'] < 1), 'New_loan_term'] = 'low'
df.loc[(df['loan_term'] >= 1) & (df['loan_term'] < 10), 'New_loan_term'] = 'normal'
df.loc[(df['loan_term'] >= 10), 'New_loan_term'] = 'high'

df["total_assets_value"]=df["bank_asset_value"]+df['luxury_assets_value']+df['residential_assets_value']+df['commercial_assets_value']
df['total_assets_value'].mean()
df['total_assets_value'] = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
df.loc[(df['total_assets_value'] < 32549622), 'Total_Assets'] = 'Poor'
df.loc[(df['total_assets_value'] >= 32549622), 'Total_Assets'] = 'Rich'

df.head()

def grab_col_names(dataframe, cat_th=11, car_th=19):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"] # 0,1,2
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"] # name
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # cat_cols tüm object  veri tipini tuttugu için içerisinde cat_but_car bulunabilir.

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat] #numerik_gorunumlu kategorikler hariç

    print(f"Observations: {dataframe.shape[0]}") # satır
    print(f"Variables: {dataframe.shape[1]}") # sutun
    print(f'cat_cols: {len(cat_cols)}') # categorik degişken sayısı
    print(f'num_cols: {len(num_cols)}') # numerik değişkenler
    print(f'cat_but_car: {len(cat_but_car)}') # categorik fakat kardinal
    print(f'num_but_cat: {len(num_but_cat)}') # numerik görünümlü kategorik

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols #loan id cıkması lazım
num_cols.remove("loan_id") #çıkarttım :)
cat_cols.append("loan_id")
cat_but_car


#############################################
# Label Encoding & Binary Encoding
#############################################


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()
df.head()
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols #hangileri binary 0 ve 1 değerini aldılar
for col in binary_cols:
    label_encoder(df, col)

df.head(200)


#############################################
# şuandaki katagorik değerlerin kendi pasta grafiği dağılımları
#  Toprak _ Melike_Ayzıt
#############################################


def approved_pie_charts(df, categorical_columns):
    approved_df = df[df[" loan_status"] == " Approved"]

    for column in categorical_columns:
        counts = approved_df[column].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(f"Pie Chart for {column}")
        plt.show()

#  Kredi verme olasılığı :
#  eğitim durumu etkilemiyor
#  self_employed etkilemiyor
#  no_of_dependents etkilemiyor
#  loan_term kısa süre ( 2-4-6 ) bi nebze etkiliyor. aslında çok da bir fark yok
#  Veri seti fazla iyi

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#Korelasyon analizi yapınız

df.plot.scatter("income_annum", "loan_amount")
plt.show()

df["income_annum"].corr(df["loan_amount"])#orta şiddetin biraz altında
#
corr = df[num_cols].corr()
corr
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr,annot=True , cmap="RdBu")
plt.show()
#
plt.figure(figsize=(8, 6))
plt.hist(df["loan_amount"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Kredi Miktarı")
plt.ylabel("Frekans")
plt.title("Kredi Miktarı Dağılımı")
plt.show()
df.head()
#
plt.figure(figsize=(6, 6))
df["loan_status"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["lightcoral", "lightgreen"])
plt.title("Kredi Onay Durumu")
plt.ylabel("")
plt.show()
df.head()
#
plt.figure(figsize=(6, 6))
df["no_of_dependents"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Ailede Bakılan Kişi Sayısı")
plt.ylabel("")
plt.show()

#
sns.set(rc={'figure.figsize':(10,5)})
bars = sns.barplot(y='education', x='loan_amount', data=df, hue='loan_status', palette= 'viridis', ci=None, edgecolor='None')
for p in bars.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height() / 2, f'{width:.2f}', ha='left', va='center')
plt.show()
#
plt.figure(figsize=(15,12))
df.boxplot(rot=90, whis=3)
plt.title('Boxplot aykırı değerler')
plt.show()
#######################logistik regresyon################
#########################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate

##########################
# Target'ın Analizi
##########################
df.head()
df["loan_status"].value_counts()

sns.countplot(x="loan_status", data=df)
plt.show()

100 * df["loan_status"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df[num_cols].describe().T


#numaric değerleri histogram çizdirmek
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if " loan_status" not in col] #bağımsız değişkenleri şeçtik


for col in cols:
     plot_numerical_col(df, col)



##########################
# Target vs Features
##########################
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "loan_status", col)
#yorumu bir tanesi için
#             cibil_score
# loan_status
#0                  703.462
#1                  429.468
#kredi cıkanların  cibil skoru daha yüksek ortalamaya sahiptir.

df.head()

######################################################
# Model & Prediction
######################################################
from sklearn.linear_model import LogisticRegression
y = df["loan_status"] #bağımlı değişken

X = df.drop(["loan_status"], axis=1) #bağımsız değişkenler

log_model = LogisticRegression().fit(X, y)

log_model.intercept_ #b0 katsayısı
log_model.coef_ #değişken ağırlıkları

y_pred = log_model.predict(X)

y_pred[0:5]# ilk 5 değere baktık tahmin edilen

y[0:5]#gerçek değerler



######################################################
# Model Evaluation
######################################################
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)



######################################################
# Model Validation: Holdout modeli ikiye böl
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["loan_status"]
X = df.drop(["loan_status"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)

y = df["loan_status"]
X = df.drop(["loan_status"], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()
X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)
################ KNN ###########################
################################################
# 3. Modeling & Prediction
################################################
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)
random_user = np.array(random_user, order='C')
knn_model.predict(random_user)

# Confusion matrix için y_pred:
y_predknn = knn_model.predict(X)

X = np.array(X, order='C')
y_predknn = knn_model.predict(X)

# AUC için y_prob:
y_probknn = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_predknn))
# acc 0.92
# f1 0.89

roc_auc_score(y, y_probknn)
# 0.97

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 0.73
# 0.59
# 0.78

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 15)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)
################################################
# Random Forests
################################################
warnings.simplefilter(action='ignore', category=Warning)


y = df["loan_status"]
X = df.drop(["loan_status"], axis=1)

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")





import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#!pip install catboost
#!pip install xgboost
#!pip install lightgbm



################################################
# 3. Modeling using CART
################################################
!pip install pydotplus
!pip install skompiler
!pip install astor
!pip install joblib
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz


y = df["loan_status"]
X = df.drop(["loan_status"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)
#değeri 1 cıktı bir gariplik var olması cok zor sorgulıcaz nasıl daha iyi değerlendirebilirim.
#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#holdout yöntemine görede başarı değerimiz 1 cıktı.
# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#gözlemliyoruz veride performansı cok yüksek cıktı ama görmediği veriyi sunuduğumuzda patladı
#train setini ezberlediği için overfitting oldu.
#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

#model başarısını arttırcaz şimdi
################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 11)}
#overfitting önüne direkt geçebilcek iki parametre bunlar max_depth ve min-samples split
#peki değerleri nasıl değerlendiriyoruz bunlarıda parametreliri cağırıp ön tanımlı değerini içinde barındırcak şekilde bir aralık verebiliriz.

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              scoring="f1",
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_best_grid.best_score_
# ön tanımlı değeri accury değeridir.
#ama yukarda deneme amaçlı scoring kodu ile ister f1 , roc_auc değeri yapabiliriz.


random = X.sample(1, random_state=45)

cart_best_grid.predict(random)


################################################
# 5. Final Model
################################################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y) #final model kurduk

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


################################################
# 6. Feature Importance #değişken önemi
################################################

cart_final.feature_importances_ #değişken önem düzeyi katsayıları

def plot_importance(model, features, num=len(X), save=False): #save kısmına true dersek cıktıyı kaydetcek
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final,X)
plot_importance(cart_final,X,num=5)
################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################


train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc",
                                           cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)


plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()




def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True) #block =True silersek iki farklı paramaetre yerine aynı grafik üzerinde cizdiriyor.



val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])



################################################
# 8. Visualizing the Decision Tree
################################################

conda install graphviz
import graphviz
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

#çalışmazsa diye bu kod veriyor.
'''def tree_graph(model, col_names, file_name):
    col_names_list = col_names.tolist()  # Index'i listeye dönüştür
    fig, ax = plt.subplots(figsize=(20, 20))  # Ayarlarınızı burada yapabilirsiniz
    tree.plot_tree(model, feature_names=col_names_list, filled=True, ax=ax)
    plt.savefig(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")'''



cart_final.get_params()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(50,40))
plot_tree(cart_final, filled=True, feature_names=X.columns, class_names=["Class1", "Class2"], rounded=True)
'''plot_tree(cart_final, filled=True, feature_names=X.columns.tolist(), class_names=["Class1", "Class2"], rounded=True)'''

plt.show()
################################################
# 9. Extracting Decision Rules
################################################

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)





