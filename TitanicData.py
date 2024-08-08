# # Pandas Alıştırmalar
import  numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#  Görev 1:  Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#  Görev 2:  Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#  Görev3:  Her birsutunaait unique değerlerin sayısını bulunuz.
#  Görev4:  pclass değişkeninin unique değerlerinin sayısını bulunuz.
#  Görev5:  pclass veparch değişkenlerinin unique değerlerinin sayısını bulunuz.
#  Görev6:  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
#  Görev7:  embarked değeri C olanların tüm bilgelerini gösteriniz.
#  Görev8:  embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#  Görev9:   Yaşı30 dan küçükvekadınolanyolcularıntümbilgilerini gösteriniz.
#  Görev10:  Fare'i 500'den büyük veyayaşı 70 den büyükyolcuların bilgilerini gösteriniz.
#  Görev 11:  Her bir değişkendeki boş değerlerin toplamını bulunuz.
#  Görev 12:  who değişkenini dataframe’den çıkarınız.
#  Görev13:  deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#  Görev14:  age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
#  Görev15:  survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#  Görev16:  30 yaşınaltında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
#  setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#  Görev17:  Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#  Görev18:  Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#  Görev19:  Günlere vetime göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#  Görev 20:  Lunch zamanına vekadınmüşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#  Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
#  Görev22:  total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#  Görev23:  total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

df = sns.load_dataset("titanic")
print(df.head())
print(df.shape)

print(df["sex"].value_counts())

print(df.nunique())
print(df["pclass"].unique())

print(df[["pclass", "parch"]].nunique())

print(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)
print(df.info())

print(df[df["embarked"] == "C"].head(10))

print(df[df["embarked"] != "S"].head(10))

print(df[df["embarked"] != "S"]["embarked"].unique())

print(df[~(df["embarked"] == "S")]["embarked"].unique())

print(df[(df["age"] < 30) & (df["sex"] == "female")].head())

print(df[(df["fare"] > 500) | (df["age"] > 70)].head())

print(df.isnull().sum())

df.drop("who", axis=1, inplace=True)


deck_mode = df["deck"].mode()[0]
df["deck"] = df["deck"].fillna(deck_mode)
print(df["deck"].isnull().sum())

age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)
print(df.isnull().sum())

print(df.groupby(["pclass", "sex"], observed=True).agg({"survived": ["sum", "count", "mean"]}))

def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x: age_30(x))
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# Tips dataset
df = sns.load_dataset("tips")
print(df.head())
print(df.shape)

print(df.groupby("time", observed=True).agg({"total_bill": ["sum", "min", "mean", "max"]}))

print(df.groupby(["day", "time"], observed=True).agg({"total_bill": ["sum", "min", "mean", "max"]}))

print(df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day", observed=True).agg({
    "total_bill": ["sum", "min", "max", "mean"],
    "tip": ["sum", "min", "max", "mean"]
}))

print(df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean())  # 17.184965034965035

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
print(df.head())

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
print(new_df.shape)
print(new_df.head())
