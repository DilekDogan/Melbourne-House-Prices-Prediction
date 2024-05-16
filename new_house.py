###    Bölüm - 1 ( Veri Temizleme)   ###

###   Açıklama
"""
Suburb:Banliyö: Banliyö
Address:Adres: Adres
Rooms:Odalar: Oda sayısı
Price:Fiyat: Avustralya doları cinsinden fiyat
Method:Yöntem: S - satılan mülk; SP - önceden satılan mülk; PI - aktarılan mülk; PN - açıklanmadan önce satıldı; SN - satıldığı açıklanmadı; Not - teklif yok; VB - satıcı teklifi; W - açık artırmadan önce geri çekilmiş; SA - açık artırmadan sonra satılır; SS - açık artırma fiyatı açıklanmadan sonra satıldı; Yok - fiyat veya en yüksek teklif mevcut değil.
Type:Tür: br - yatak odası(ları); h - ev,yazlık,villa, yarı,teras; u - birim, çift yönlü; t - şehir evi; geliştirme sitesi - geliştirme sitesi; o res - diğer konut.
SellerG:SatıcıG: Emlakçı
Date: Tarih: Satış tarihi
Distance: Mesafe: Kilometre cinsinden CBD'den uzaklık
Regionname: Bölge adı: Genel Bölge (Batı, Kuzey Batı, Kuzey, Kuzey doğu…vb)
Propertycount:Mülk sayısı:: Banliyöde bulunan mülklerin sayısı.
Yatak Odası2: Kazınmış Yatak Odası Sayısı (farklı kaynaktan)
Bathroom: Banyo: Banyo Sayısı
Araba: Araç park yeri sayısı
Landsize:Arazi Büyüklüğü: Metre Cinsinden Arazi Büyüklüğü
BuildingArea: Metre cinsinden Bina Boyutu
Yapım Yılı: Evin inşa edildiği yıl
CouncilArea:Konsey Alanı:Bölgenin yönetim konseyi
Lattitude:Enlem
Longtitude: Boylam
"""
###  Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimpy import clean_columns
import scipy.stats as stats

# !pip install termcolor
import colorama
from colorama import Fore, Style  # makes strings colored
from termcolor import colored
from termcolor import cprint

# ml algorithm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, validation_curve
from yellowbrick.model_selection import ValidationCurve
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


from scipy.stats import skew


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.set_option("display.width", 500)

dff = pd.read_csv("datasets/Melbourne_housing_FULL.csv")
df = dff.copy()

df.columns
##### Bölüm 1- Veri İnceleme (EDA)
#her sütundaki eksik değerlerin sayısını ve yüzdesini hesaplar
def missing_values(df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values[missing_values['Missing_Number']>0]

missing_values(df)
####
#Genel Bakış
                                                                                                          
def first_look(df, col):
    print("column name    : ", col)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col].isnull().sum() * 100 / df.shape[0], 2))
    print("num_of_nulls   : ", df[col].isnull().sum())
    print("num_of_uniques : ", df[col].astype(str).nunique())
    print("shape_of_df    : ", df.shape)
    print("--------------------------------")
    print(df[col].value_counts(dropna=False))

    print("#####################################")
# Her değişkeni gösterir

for col in df.columns:
    if col in df.columns:
        first_look(df, col)

first_look(df, "Price")

####

def remove_duplicates(df):
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df.drop_duplicates(inplace=True)
remove_duplicates(df)
df.duplicated().sum()
#####
def show_unique_values(data, column_name=None):
    if isinstance(data, pd.DataFrame):
        if column_name:
            if column_name in data.columns:
                unique_values = data[column_name].unique()
                print(f"{column_name} sütunu için unique değerler:")
                print(unique_values)
                print(len(unique_values), "adet unique değer vardır.")
                print("------------------------")
            else:
                print(f"{column_name} sütunu DataFrame içinde bulunmuyor.")
        else:
            for column in data.columns:
                unique_values = data[column].unique()
                print(f"{column} sütunu için unique değerler:")
                print(unique_values)
                print(len(unique_values), "adet unique değer vardır.")
                print("------------------------")
    else:
        print("Lütfen bir DataFrame girin.")


show_unique_values(df)
show_unique_values(df, "Car")
####
df. groupby("Regionname")["Price"].mean()

#### Veri Görselleştirme

#Melbourne  Housing Map
#Her bir nokta, bir evin konumunu temsil eder. Noktaların renkleri evlerin fiyatını gösterir;
#daha yüksek fiyatlar daha sıcak renklerle (örneğin kırmızı) gösterilir.
plt.figure(figsize= (8,8))
plt.scatter(df['Lattitude'] , df['Longtitude'], c=df['Price'],cmap='hot')
plt.colorbar()
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Melbourne  Housing Map")

#her bir oda sayısına sahip evlerin sayısını gösterir.
df.groupby(["Rooms"])['Price'].count().plot(kind="bar")
plt.xlabel("Number of Rooms")
plt.ylabel("Count of item")
plt.title("Count of prices by number of room")
plt.show()
#####

#h - ev,yazlık,villa, yarı,teras; ,u - birim, çift yönlü; , t - şehir evi;
light_palette = "pastel"
plt.figure(figsize=(12, 8))
# Kutu grafiği
sns.boxplot(x='Regionname', y='Price', hue='Type', data=df, palette=light_palette, linewidth=1)

plt.title('Distribution of Prices by Regionname and Type')
plt.xlabel('Regionname')
plt.ylabel('Price')
plt.xticks(rotation=15)

plt.legend(title='Type')
plt.show()
####
#Bu grafik, farklı bölgelerdeki ev fiyatlarının ortalamasını gösterir.
df.groupby(["Regionname"])['Price'].mean().plot(kind='bar')
plt.xlabel("Regionname")
plt.ylabel("mean of Price")
plt.title("Total Price by Region")
plt.xticks(rotation=15, ha='right')

plt.show()
####
#Bu grafik, farklı bölgelerdeki evlerin toplam satış sayılarını karşılaştırarak
#hangi bölgelerin daha çok veya daha az satılan evlere sahip olduğunu gösterir.

plt.figure(figsize=(10, 8))
ax = df.groupby(["Regionname"]).size().plot(kind='bar')
plt.xlabel("Regionname")
plt.ylabel("Total Sales")
plt.title("Total Sales by Region")
plt.xticks(rotation=25, ha='right')

# Her bir çubuğun üstüne sayıları ekler
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()
######

def visualize_data(df, column, plot_type='histogram', iqr_multiplier=1.5, **kwargs):
    data = df[column]

    if plot_type == 'all':
        plot_types = ['histogram', 'box', 'qq']
        for plot_type in plot_types:
            visualize_data(df, column, plot_type=plot_type, iqr_multiplier=iqr_multiplier, **kwargs)
    else:
        if plot_type == 'histogram':
            plt.figure(figsize=(10, 6))
            sns.histplot(data, kde=True, **kwargs)

            skewness = data.skew()
            """

data.skew() bir veri kümesinin çarpıklık (skewness) değerini hesaplar. 
Çarpıklık, veri dağılımının simetrisizliğini ölçer. Bir veri dağılımı simetrik ise 
çarpıklık değeri sıfırdır. Pozitif çarpıklık, dağılımın sağa doğru uzandığını (sağa çarpık) 
ve negatif çarpıklık ise dağılımın sola doğru uzandığını (sola çarpık) gösterir. 
Yani, data.skew() çağrısı, veri setinin çarpıklık ölçüsünü döndürür. 
Bu değer, veri dağılımının ne kadar simetrik veya asimetrik olduğunu belirtir.
            """
            kurtosis = data.kurtosis()
            mean = data.mean()
            median = data.median()
            mode = data.mode().iloc[0]
            # Başlığı ve değerleri düzenleyelim
            title = f'Histogram of {column}\n\n'
            title += f'Skewness: {skewness:.2f}, '
            title += f'Kurtosis: {kurtosis:.2f}\n'
            title += f'Mean: {mean:.2f}, '
            title += f'Median: {median:.2f}, '
            title += f'Mode: {mode:.2f}'
            plt.title(title, fontweight='bold')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        elif plot_type == 'box':

            mean = np.mean(data)
            median = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            upper_whisker = q3 + iqr_multiplier * iqr
            lower_whisker = q1 - iqr_multiplier * iqr
            min_whisker = np.min(data[data >= lower_whisker])
            max_whisker = np.max(data[data <= upper_whisker])

            num_below_min_whisker = np.sum(data < lower_whisker)
            num_above_max_whisker = np.sum(data > upper_whisker)
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=data, **kwargs)
            plt.title(
                f'Box Plot of {column}\n\nMean: {mean:.2f}, Median: {median:.2f}, Min Whisker: {min_whisker:.2f}, Max Whisker: {max_whisker:.2f}\nIQR Multiplier: {iqr_multiplier}\n\nNumber of Extreme Values in Below Min Whisker: {num_below_min_whisker}\n Number of Extreme Values in Above Max Whisker: {num_above_max_whisker}',
                fontweight="bold")
            plt.ylabel(column)
            plt.show()
        elif plot_type == 'qq':

            plt.figure(figsize=(10, 6))
            stats.probplot(data, dist="norm", plot=plt)
            plt.title(f'Q-Q Plot of {column}\nNormality Check', fontweight='bold')
            plt.xlabel('Theoretical Quantiles')
            plt.ylabel('Sample Quantiles')
            plt.show()
        elif plot_type == 'bar':
            value_counts = data.value_counts()
            x = value_counts.index
            y = value_counts.values
            total_height = sum(y)
            percentages = [(count / total_height) * 100 for count in y]
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(x, percentages, **kwargs)  # Horizontal bar chart
            plt.title(f'Bar Chart of {column}', fontweight='bold')
            plt.xlabel('Percentage')
            plt.ylabel(column)
            for bar, percentage in zip(bars, percentages):
                width = bar.get_width()
                ax.annotate(f'{percentage:.2f}%', xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0), textcoords='offset points', va='center')
            plt.show()
        elif plot_type == 'pie':
            value_counts = data.value_counts()
            labels = value_counts.index
            sizes = value_counts.values
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, **kwargs)
            plt.title(f'Pie Chart of {column}', fontweight='bold')
            plt.axis('equal')
            plt.show()


visualize_data(df,column="Price", plot_type="all")


#####
df.nunique()
def scatter_or_joint_plot(df, x_col, y_col, title=None, x_label=None, y_label=None, kind='scatter', **kwargs):
    if title is None:
        title = f'Plot of {x_col} vs {y_col}'
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col

    plt.figure(figsize=(10, 6))
    corr_coefficient = np.corrcoef(df[x_col], df[y_col])[0, 1]

    if kind == 'scatter':
        sns.scatterplot(data=df, x=x_col, y=y_col, **kwargs)
        title_with_corr = f'{title} (Scatter)\n\nCorrelation: {corr_coefficient:.2f}'
        plt.title(title_with_corr, fontweight='bold')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
    elif kind == 'joint':
        g = sns.jointplot(x=x_col, y=y_col, data=df, **kwargs)
        g.fig.suptitle(f'{title}', fontweight='bold', y=1.02)
        g.ax_joint.set_xlabel(x_label)
        g.ax_joint.set_ylabel(y_label)
        g.ax_marg_x.set_title(f'Correlation: {corr_coefficient:.2f}', fontweight='bold')
        plt.subplots_adjust(top=0.9)

    plt.show()

scatter_or_joint_plot(df, "Rooms", "Price", title=None, x_label="Rooms", y_label="Price", kind='scatter')
####
#Categorical or Numerical Seperation (Kategorik ve Numerik Ayırma)

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 250  and df[col].dtype in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if
          df[col].nunique() > 500 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]
####
#Numerik değişkenleri grafik ile inceleme:
def num_summary(dataframe, numerical_col, plot=False, ax=None):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[[numerical_col]].describe(quantiles).T)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))  # Tek bir tablo için figür ve eksen oluştur
        dataframe[numerical_col].hist(bins=20, ax=ax)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(numerical_col, fontsize=14)

# Grafikleri tek bir tablo içinde göstermek için plt.show() fonksiyonunu burada çağırmıyoruz
fig, axs = plt.subplots((len(num_cols) + 3) // 4, 4, figsize=(20, (len(num_cols) + 3) * 3))

for i, col in enumerate(num_cols):
    num_summary(df, col, plot=True, ax=axs[i // 4, i % 4])      #sütun ve satır index
plt.show()
####


###    Veriyi Okuma ve Anlama   ###

missing_values(df)
df.duplicated().sum()

####
#### Bölüm - 2 Özellik Mühendisliği


#### Boş değerleri doldurma

def fills_with_mode(df, group_col1, group_col2, col_name):

    group1_modes = df.groupby(group_col1)[col_name].transform(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    group2_modes = df.groupby([group_col1, group_col2])[col_name].transform(
        lambda x: x.mode().iat[0] if not x.mode().empty else None)
    global_mode = df[col_name].mode().iat[0] if not df[col_name].mode().empty else None

    for index, row in df.iterrows():
        if pd.notna(row[col_name]):
            continue
        group1_val = row[group_col1]
        group2_val = row[group_col2]
        if pd.notna(group2_modes[index]):
            df.at[index, col_name] = group2_modes[index]
        elif pd.notna(group1_modes[index]):
            df.at[index, col_name] = group1_modes[index]
        else:
            df.at[index, col_name] = global_mode

    print("column name    : ", col_name)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col_name].isnull().sum() * 100 / df.shape[0], 2))
    print("num_of_nulls   : ", df[col_name].isnull().sum())
    print("num_of_uniques : ", df[col_name].nunique())
    print("--------------------------------")
    print(df[col_name].value_counts(dropna=False))


def fills(df, group_col1, group_col2, col_name, method):

    if method == "mode":
        for group1 in list(df[group_col1].unique()):
            for group2 in list(df[group_col2].unique()):
                cond1 = df[group_col1] == group1
                cond2 = (df[group_col1] == group1) & (df[group_col2] == group2)
                mode1 = list(df[cond1][col_name].mode())
                mode2 = list(df[cond2][col_name].mode())
                if mode2 != []:
                    df.loc[cond2, col_name] = df.loc[cond2, col_name].fillna(df[cond2][col_name].mode()[0])
                elif mode1 != []:
                    df.loc[cond2, col_name] = df.loc[cond2, col_name].fillna(df[cond1][col_name].mode()[0])
                else:
                    df.loc[cond2, col_name] = df.loc[cond2, col_name].fillna(df[col_name].mode()[0])

    elif method == "mean":
        df[col_name].fillna(df.groupby([group_col1, group_col2])[col_name].transform("mean"), inplace=True)
        df[col_name].fillna(df.groupby(group_col1)[col_name].transform("mean"), inplace=True)
        df[col_name].fillna(df[col_name].mean(), inplace=True)

    elif method == "median":
        df[col_name].fillna(df.groupby([group_col1, group_col2])[col_name].transform("median"), inplace=True)
        df[col_name].fillna(df.groupby(group_col1)[col_name].transform("median"), inplace=True)
        df[col_name].fillna(df[col_name].median(), inplace=True)

    elif method == "ffill":
        for group1 in list(df[group_col1].unique()):
            for group2 in list(df[group_col2].unique()):
                cond2 = (df[group_col1] == group1) & (df[group_col2] == group2)
                df.loc[cond2, col_name] = df.loc[cond2, col_name].fillna(method="ffill").fillna(method="bfill")

        for group1 in list(df[group_col1].unique()):
            cond1 = df[group_col1] == group1
            df.loc[cond1, col_name] = df.loc[cond1, col_name].fillna(method="ffill").fillna(method="bfill")

        df[col_name] = df[col_name].fillna(method="ffill").fillna(method="bfill")

    print("COLUMN NAME    : ", col_name)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col_name].isnull().sum() * 100 / df.shape[0], 2))
    print("num_of_nulls   : ", df[col_name].isnull().sum())
    print("num_of_uniques : ", df[col_name].nunique())
    print("--------------------------------")
    print(df[col_name].value_counts(dropna=False).sort_index())



### korelasyon kontrol


def corr(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numeric_columns].corrwith(df[col_name])
    print(correlations)

corr(df,'Price')
df.shape
###
"""def categorical_numeric_relationship(df, cat_col, num_col):
    # Kategorik değişkenin her bir kategorisi için, nümerik değerlerin ortalaması
    cat_groups = df.groupby(cat_col)[num_col].mean()
    print(cat_groups)
categorical_numeric_relationship(df, 'Propertycount', 'Type')
"""



###

visualize_data(df, 'Price', 'histogram')
first_look(df, 'Price')
corr(df, 'Price')
fills(df, 'Rooms', 'Type', 'Price', 'median')

#Type ve rooms un boş değeri yok . ayrıca price ın en yüksek korelasyonu rooms ile.
missing_values(df)

####

visualize_data(df, 'BuildingArea', 'histogram')
first_look(df, 'BuildingArea')
corr(df, 'BuildingArea')
fills(df, 'Rooms', 'Price', 'BuildingArea', 'median')

####
visualize_data(df, 'Landsize', 'histogram')
first_look(df, 'Landsize')
corr(df, 'Landsize')
fills(df, 'Rooms', 'BuildingArea', 'Landsize', 'median')

####
visualize_data(df, 'Bathroom', 'histogram')
first_look(df, 'Bathroom')
corr(df, 'Bathroom')
fills(df, 'Rooms', 'Price', 'Bathroom', 'median')

####
visualize_data(df, 'Bedroom2', 'histogram')
first_look(df, 'Bedroom2')
corr(df, 'Bedroom2')
fills(df, 'Rooms', 'Bathroom', 'Bedroom2', 'median')

####
visualize_data(df, 'Distance', 'histogram')
first_look(df, 'Distance')
corr(df, 'Distance')
fills(df, 'YearBuilt', 'Postcode', 'Distance', 'median')

####
visualize_data(df, 'Car', 'histogram')
first_look(df, 'Car')
corr(df, 'Car')
fills(df, 'BuildingArea', 'Landsize', 'Car', 'median')
"""
BuildingArea: Evlerin inşaat alanı kapalı alanın büyüklüğünü temsil eder ve 
bu da park yeri sayısını etkileyebilir.

Landsize: Evlerin arazi büyüklüğü park yeri sayısını etkileyebilir, çünkü arazi büyüklüğü 
daha büyük olan evler daha fazla park alanına sahip olabilir.
"""
####

visualize_data(df, 'Propertycount', 'histogram')
first_look(df, 'Propertycount')
corr(df, 'Propertycount')
fills_with_mode(df, 'Suburb', 'Postcode', 'Propertycount')
####
visualize_data(df, 'YearBuilt', 'histogram')
first_look(df, 'YearBuilt')
corr(df, 'YearBuilt')
fills_with_mode(df, 'Suburb', 'Distance', 'YearBuilt')

####
visualize_data(df, 'Regionname', 'histogram')
first_look(df, 'Regionname')
fills_with_mode(df, 'Suburb', 'Postcode', 'Regionname')

####

visualize_data(df, 'CouncilArea', 'histogram')
first_look(df, 'CouncilArea')
corr(df, 'CouncilArea')
fills_with_mode(df, 'Suburb', 'Postcode', 'CouncilArea')

####
df.shape
df.isnull().sum()
####  Aykırı Değerlere Çözüm
df.reset_index(drop=True, inplace=True)
df.info()


####
## bağımlı değişken inceleme
first_look(df,'Price')
df.Price.describe().T
visualize_data(df, "Price", plot_type="all")
df.sort_values(by=["Price"], ascending=False)["Price"].head(20)
df[df["Price"] == 11200000.0].T
df.sort_values(by=["Price"], ascending=True)["Price"].head(20)

#### Outliers için  Fonksiyonlar

def drop_outliers_zscore(df, feature, threshold=3):

   print(f"Number of rows before dropping outliers from {feature}:", len(df))
   z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
   df = df[z_scores < threshold]
   df.reset_index(drop=True, inplace=True)
   print(f"Number of rows after dropping outliers from {feature}:", len(df))

   return df


def drop_outliers_zscore_for_all(df, threshold=3):
    excluded_columns = ["Postcode", "Lattitude", "Longtitude"]
    temp_df = df.drop(excluded_columns, axis=1)

    numeric_columns = temp_df.select_dtypes(include=['number']).columns

    for column in numeric_columns:
        df = drop_outliers_zscore(df, column, threshold)

    return df


df = drop_outliers_zscore_for_all(df, threshold=3)
df.info()
df.shape #(31611, 21)


################

df["building_age"] = 2023 - df["YearBuilt"]
df.drop("YearBuilt", axis=1, inplace=True)

df["total_room_bathroom"] = df["Rooms"] + df["Bathroom"]
df["total_room_bathroom"].value_counts()

df['total_area'] = df['Landsize'] + df['BuildingArea']
df["total_area"].value_counts().sort_index()

##################

####   Bölüm 3 - Makine Öğrenmesi
#veri sızıntısı olmaması için duplicate i düşürmemiz gerek, kontrol edelim
df.duplicated().sum()
#kullanmadığıklarımızı düşürelim
df.drop(["Postcode", "Lattitude", "Longtitude", "Address", "Date"], axis = 1, inplace = True)

sifirlar= (df == 0).sum()


#Sıfırları anlamlı hale getirelim:

def height_solver(df, how='likeZero'):
    if how == 'likeZero':
        zero_columns = df.columns[df.eq(0).any()]
        df[zero_columns] = df[zero_columns].replace(0, 0.0001)
        return df

    elif how == 'knnImputer':
        zero_columns = df.columns[df.eq(0).any()]
        df[zero_columns] = df[zero_columns].replace(0, np.nan)

        imputer = KNNImputer(n_neighbors=5)
        df[zero_columns] = imputer.fit_transform(df[zero_columns])
        return df

height_solver(df, how='likeZero')

### Multicollinearity Kontrolü ( )

df.select_dtypes(include="number")
df_numeric = df.select_dtypes(include ="number")
plt.figure(figsize=(12,8))
sns.heatmap(df_numeric.corr(), annot=True , vmin=-1 , vmax=1, cmap="coolwarm")
####
# aralarında yüksek korelasyon var mı
df_numeric.corr()[(abs(df_numeric.corr()) >= 0.9) & (abs(df_numeric.corr()) < 1)].any()

### Multicolliniearity için VIF Skoru
#yüksek korelasyon ölmek için,korelasyon matrisi
##VIF (Variance Inflation Factor), çoklu kolinearitenin varlığını belirlemek için kullanılan bir istatistiksel ölçüdür.
# vif = 1/1-rkare
X_vif = df_numeric.drop(columns='Price')
X_vif.head()

df_vif = pd.DataFrame()
df_vif['features'] = X_vif.columns

variance_inflation_factor(X_vif.values, 0)
df_vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

###
df_vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

df_vif

### Train | Test Split

df_new = df.copy()
X = df_new.drop(columns="Price")
y = df_new["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=42)

X_train.head()

### Objectleri Sayısal Sütuna Çevirme, makine öğrenmesi için analiz
df_object = df_new.select_dtypes(include ="object")
for col in df_object:
    print(f"{col:<30}:", df[col].nunique())

#one hot enc yaparsak suburb 301 sütun olcak demek

cat_onehot = ['Type']
cat_ordinal_encoder = ['Suburb', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
df.shape
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_onehot),
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_ordinal_encoder),
    remainder='passthrough',
    verbose_feature_names_out=False
)

column_trans = column_trans.set_output(transform="pandas")

X_train_trans = column_trans.fit_transform(X_train)
X_test_trans = column_trans.transform(X_test)

X_train_trans.shape, X_test_trans.shape

X_train_trans
X_test_trans
#sütun seçelim 19 sütun değil de 3 alalım sütun seçme işlemi yapalım,
#scaling işlemi yapmazsak model bizi yanıltabiliyor.Range daha yüksek olan sütunun daha etkili old söyleyebiliyor.
# O yüzden mach learning modelimize datamızı eklemeden önce scaler yapmamız gerekiyor. burada min max scaler seçtik

###Scaling:
"""" 
-MinMax Scaling
-Robust Scaler
-MaxAbs Scaler
-Standardizasyon
-PowerTransformer

"""

scaler = MinMaxScaler().set_output(transform="pandas")
scaler.fit(X_train_trans)
# dogru feature selection yapmak icin butun featurelari ayni araliga almamiz lazım
X_train_scaled = scaler.transform(X_train_trans)
X_test_scaled = scaler.transform(X_test_trans)


scaler = MinMaxScaler()

X_train_scaled.head()
X_test_scaled.head()

### Linear Regresyon Uygulaması

def train_val(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    scores = {"train": {"R2": r2_score(y_train, y_train_pred),
                        "mae": mean_absolute_error(y_train, y_train_pred),
                        "mse": mean_squared_error(y_train, y_train_pred),
                        "rmse": mean_squared_error(y_train, y_train_pred, squared=False),
                        "mape": mean_absolute_percentage_error(y_train, y_train_pred)},

              "test": {"R2": r2_score(y_test, y_pred),
                       "mae": mean_absolute_error(y_test, y_pred),
                       "mse": mean_squared_error(y_test, y_pred),
                       "rmse": mean_squared_error(y_test, y_pred, squared=False),
                       "mape": mean_absolute_percentage_error(y_test, y_pred)}}

    return pd.DataFrame(scores)

lm = LinearRegression()
lm.fit(X_train_scaled, y_train)

pd.options.display.float_format = '{:.3f}'.format
train_val(lm, X_train_scaled, y_train, X_test_scaled, y_test)
###
##Cross Validate(Çapraz Doğrulama)
#seçtiğimiz random_state doğru seçim mi,doğru yerden mi böldü çapraz doğrulama ile kontrol edelim:
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Linear", LinearRegression())]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(pipe_model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv=10,
                        return_train_score=True)

###
scores = pd.DataFrame(scores, index = range(1, 11))
scores

"""
Bu sonuçlar, modelin makul bir performans sergilediğini ancak hala iyileştirme alanları 
bulunduğunu göstermektedir. Özellikle, modelin aşırı uyum (overfitting) yapma potansiyeli 
olduğunu ve belirli hata metriklerinde iyileştirme yapılabilir olduğunu görebiliriz.
"""
####
scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
#benzer skorlar gördük meanler yakın overfit yok

## Pipeline

X = df_new.drop(columns=["Price"])
y = df_new.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.head()

cat_onehot = ['Type']
cat_ordinal_encoder = ['Suburb', 'Method', 'SellerG', 'CouncilArea', 'Regionname']


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_onehot),
    (OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1), cat_ordinal_encoder),
    remainder='passthrough',
    verbose_feature_names_out=False
)

##Ridge
# linear reg bir türü
# Ridge regresyon modeli, özellikle büyük özellik setleriyle çalışırken, aşırı uyumun önlenmesi
# ve modelin daha iyi performans göstermesi için kullanışlıdır.
# ! araştır doğru mu: overfit i ortadan kaldırmak için
column_trans = column_trans.set_output(transform="pandas")

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Ridge", Ridge())]

ridge_model = Pipeline(steps=operations).set_output(transform="pandas")
# If we want the outputs of the given transform algorithms to be dataframes,
# you can add set_output(transform="pandas") to the end of the pipeline.

ridge_model.fit(X_train, y_train)

train_val(ridge_model, X_train, y_train, X_test, y_test)

### Çapraz Doğrulama

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Ridge", Ridge())]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(pipe_model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv=10,
                        return_train_score=True)

scores = pd.DataFrame(scores, index = range(1, 11))
scores
##
scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()

##

## Ridge için en iyi alfayı bulma

alpha_space = np.linspace(-1, 100, 100)

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Ridge", Ridge())]

pipe_model = Pipeline(steps=operations)



param_grid = {'Ridge__alpha':alpha_space}  # # Parameter names should be used together with the model name defined
                                           # in the pipeline..

ridge_grid_model = GridSearchCV(estimator=pipe_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1,
                          return_train_score=True)

pipe_model.get_params()
##
ridge_grid_model.fit(X_train, y_train)
##
ridge_grid_model.best_estimator_
##
pd.DataFrame(ridge_grid_model.cv_results_).loc[ridge_grid_model.best_index_, ["mean_test_score", "mean_train_score"]]
##
train_val(ridge_grid_model, X_train, y_train, X_test, y_test)
##
y_pred = ridge_grid_model.predict(X_test)
rm_R2 = r2_score(y_test, y_pred)
rm_mae = mean_absolute_error(y_test, y_pred)
rm_rmse = mean_squared_error(y_test, y_pred, squared=False)
rm_mape= mean_absolute_percentage_error(y_test, y_pred)
##
ridge_grid_model.best_estimator_["Ridge"].coef_
##
ridge_grid_model.best_estimator_["OneHot_Encoder"].get_feature_names_out()
##
pd.DataFrame(data= ridge_grid_model.best_estimator_["Ridge"].coef_,
             index=ridge_grid_model.best_estimator_["OneHot_Encoder"].get_feature_names_out(),
             columns=["Coef"]).sort_values("Coef")

## Lasso Regresyon
"""operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Lasso", Lasso())]

lasso_model = Pipeline(steps=operations)

lasso_model.fit(X_train, y_train)
##

train_val(lasso_model, X_train, y_train, X_test, y_test)
##

##Çapraz Doğrulama
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Lasso", Lasso())]

pipe_model = Pipeline(steps=operations)
scores = cross_validate(pipe_model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv=10,
                        return_train_score=True)

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()"""
##
"""## Lasso için en iyi alpha bulma
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("Lasso", Lasso())]

model = Pipeline(steps=operations)

param_grid = {'Lasso__alpha':alpha_space}# Parameter names should be used together with the model name defined in the pipeline.

lasso_grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1,
                          return_train_score=True)

lasso_grid_model.fit(X_train, y_train)

lasso_grid_model.best_estimator_
##
pd.DataFrame(lasso_grid_model.cv_results_).loc[lasso_grid_model.best_index_, ["mean_test_score", "mean_train_score"]]

##
train_val(lasso_grid_model, X_train, y_train, X_test, y_test)
##
y_pred = lasso_grid_model.predict(X_test)
lasm_R2 = r2_score(y_test, y_pred)
lasm_mae = mean_absolute_error(y_test, y_pred)
lasm_rmse = mean_squared_error(y_test, y_pred, squared=False)
lasm_mape= mean_absolute_percentage_error(y_test, y_pred)
##
pd.DataFrame(data=lasso_grid_model.best_estimator_["Lasso"].coef_,
             index=lasso_grid_model.best_estimator_["OneHot_Encoder"].get_feature_names_out(),
             columns=["Coef"]).sort_values("Coef")
##

##Elastic-Net Uygulaması

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("ElasticNet", ElasticNet())]

elastic_model = Pipeline(steps=operations)

elastic_model.fit(X_train, y_train)

train_val(elastic_model, X_train, y_train, X_test, y_test)

## Çapraz Doğrulama

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("ElasticNet", ElasticNet())]

model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv=10,
                        return_train_score=True)

##
scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
##
## Elastik-Net için en iyi alpha ve l1_ratio bulma

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("ElasticNet", ElasticNet())]

model = Pipeline(steps=operations)

param_grid = {'ElasticNet__alpha':np.linspace(-1, 1, 15), #[0.001,0.01, 0.5, 1, 2,  3, 4],
              'ElasticNet__l1_ratio':[0.5, 0.7, 0.9, 0.95, 0.99, 1]}

elastic_grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1,
                          return_train_score=True)

##
elastic_grid_model.fit(X_train, y_train)
##
elastic_grid_model.best_estimator_
##
pd.DataFrame(elastic_grid_model.cv_results_).loc[elastic_grid_model.best_index_, ["mean_test_score", "mean_train_score"]]
##
train_val(elastic_grid_model, X_train, y_train, X_test, y_test)
##
y_pred = elastic_grid_model.predict(X_test)
em_R2 = r2_score(y_test, y_pred)
em_mae = mean_absolute_error(y_test, y_pred)
em_rmse = mean_squared_error(y_test, y_pred, squared=False)
em_mape= mean_absolute_percentage_error(y_test, y_pred)
##
## Feature önemi/seçimi

df_feat_imp =pd.DataFrame(
                         data=lasso_grid_model.best_estimator_["Lasso"].coef_,
                         index=lasso_grid_model.best_estimator_["OneHot_Encoder"].get_feature_names_out(),
                         columns=["Coef"]
                         ).sort_values("Coef")## Feature İmportance/selection
##
df_feat_imp
##
plt.figure(figsize=(10,14))
sns.barplot(data= df_feat_imp,
            x=df_feat_imp.Coef,
            y=df_feat_imp.index);
##
#Olumlu etkiye sahip en önemli iki özellik bina alanıdır, mesafe ise olumsuz etkiye sahiptir.
lasso_grid_model.best_estimator_["Lasso"]

model = lasso_grid_model.best_estimator_["Lasso"] # Lasso(alpha=0.001)

viz = FeatureImportances(model,
                         labels=lasso_grid_model.best_estimator_["OneHot_Encoder"].get_feature_names_out())

visualizer = RadViz(size=(720, 3000))
viz.fit(X_train, y_train)
viz.show()
##
## Karar Ağacı Regresyon
#Pipeline
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(random_state=42))]

decision_model = Pipeline(steps=operations)

decision_model.fit(X_train, y_train)

#
train_val(decision_model, X_train, y_train, X_test, y_test)
#
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(random_state=42))]
model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv =10,
                        return_train_score=True)

df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]

##
visualizer = RadViz(size=(720, 600))

model = decision_model
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
visualizer.show()
##
## Grid Search
param_grid = {"DT_model__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
              "DT_model__ccp_alpha": [.04, .043, .05],
              "DT_model__max_depth": [None, range(2, 15, 1)]}
##
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(random_state=42))]

pipe_model = Pipeline(steps=operations)
grid_model = GridSearchCV(estimator=pipe_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs=-1,
                          return_train_score=True)
##
grid_model.fit(X_train,y_train)
##
grid_model.best_estimator_
##
pd.DataFrame(grid_model.cv_results_).loc[grid_model.best_index_, ["mean_test_score", "mean_train_score"]]
##
train_val(grid_model, X_train, y_train, X_test, y_test)
##
from sklearn.model_selection import cross_validate, cross_val_score

operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(ccp_alpha=0.04,
                                                 criterion='friedman_mse',
                                                 random_state=42,
                                                 max_depth =8 ))]

model = Pipeline(steps=operations)


scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=['r2',
                                 'neg_mean_absolute_error',
                                 'neg_mean_squared_error',
                                 'neg_root_mean_squared_error',
                                 'neg_mean_absolute_percentage_error'],
                        cv = 10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index=range(1,11))
df_scores.iloc[:,2:]
##
df_scores.mean()[2:]
##
##Feature Importance
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(ccp_alpha=0.04,
                                                 criterion='friedman_mse',
                                                 random_state=42,
                                                 max_depth =8 ))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)
##
X_train.head(1)
features = pipe_model["OneHot_Encoder"].get_feature_names_out()

features
#
pipe_model["OneHot_Encoder"].fit_transform(X_train).head()
##
df_f_i = pd.DataFrame(data = pipe_model["DT_model"].feature_importances_,
                      index=features,
                      columns=["Feature Importance"])

df_f_i = df_f_i.sort_values("Feature Importance", ascending=False)

df_f_i
##
ax = sns.barplot(x = df_f_i.index, y = 'Feature Importance', data = df_f_i)
ax.bar_label(ax.containers[0],fmt="%.3f" ,rotation = 90)
plt.xticks(rotation = 90)
plt.tight_layout()
##
#Gerçek ve tahmini sonuçları karşılaştırma

y_pred = grid_model.predict(X_test)
my_dict = { 'Actual': y_test, 'Pred': y_pred, 'Residual': y_test-y_pred }
compare = pd.DataFrame(my_dict)
##
comp_sample = compare.sample(20)
comp_sample
##
comp_sample.plot(kind='bar',figsize=(9,5))
plt.show()
##
## Final Model
X=df_full_new.drop("price", axis=1)
y=df_full_new.price
##
operations = [("OneHot_Encoder", column_trans),
              ("scaler", scaler),
              ("DT_model", DecisionTreeRegressor(ccp_alpha=0.04,
                                                 criterion='friedman_mse',
                                                 random_state=42,
                                                 max_depth =8 ))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X, y)"""
##
## Özet
"""
Log dönüşümü yapılmadan, verilerin doğrusal regresyona uygun olmadığı açıktır ve bu aynı zamanda verilerin doğası gereği doğrusal olmadığını da gösterir.

Doğrusal modellerde R-kare puanının düşük olması, doğrusal regresyonun uygun bir seçim olmadığını göstermektedir. Bununla birlikte, log dönüşümünden sonra R-kare dışındaki diğer metriklerdeki iyileşme, model performansında önemli bir iyileşme olduğunu göstermektedir.

Karar Ağacı modeli gibi doğrusal olmayan modellerde metriklerin artması, verilerin doğrusal olmayan özelliklerini daha da vurgulamaktadır. Random Forest veya XGBoost gibi modelleri kullanarak daha da iyi puanlar elde etmeniz muhtemeldir.

Verilerde mevcut olan çoklu doğrusallığın etkilerini azaltmak için sırt ve kement gibi düzenlileştirme yöntemleri kullanılmıştır. Bilindiği gibi düzenlileştirme, potansiyel aşırı uyumu azaltmak ve varyans ile önyargı arasında dengeli bir denge sağlamak için verilere bir hata terimi ekler.

Verilerdeki doğrusal olmayan ilişkiler nedeniyle ağaç tabanlı modellerle üstün sonuçlar elde edilmiştir. Analiz, doğrusal, sırt, kement, elastik ağ ve karar ağacı dahil olmak üzere çeşitli modellerin uygulanmasını içeriyordu. Sonuçta karar ağacı modeli en iyi sonuçları verdi.
"""







