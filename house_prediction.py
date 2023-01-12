from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, linear_model
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import re
#設定中文
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["font.size"] = 15
#讀取資料，內容為經過預處理106-111年的房價實價登錄資料
data = pd.read_csv("111-106-4.csv")

#%% 整理
#自訂函數 將x欄位的值依照有無改成(1,0)
def tr(x):
    x = preprocessing.LabelEncoder().fit_transform(x)
    x = np.where(x == 1,0,1)
    return x
#運用自訂函數 將x欄位的值依照有無改成(1,0)
com = tr(data["建物現況格局-隔間"])
data["建物現況格局-隔間"] = com
#運用自訂函數 將x欄位的值依照有無改成(1,0)
man = tr(data["有無管理組織"])
data["有無管理組織"] = man
#將有車位設成1 沒車位設成0
park = preprocessing.LabelEncoder().fit_transform(data["車位類別"])
park  = np.where(park == 7,0,1)
data["車位類別"] = park

#建立將中文數字改成阿拉伯數字的轉換表
cn_to_num={"一":"1","二":"2","三":"3","四":"4","五":"5"
           ,"六":"6","七":"7","八":"8","九":"9","十":"10"}
#將中文數字改成阿拉伯數字
con = 0
for i in data["移轉層次"]:    
    # print(i)
    if len(i) == 2:
        i = cn_to_num[i[0]]
        # print(i)
        data["移轉層次"][con] = i
        con += 1        
    elif len(i) == 3:
        if i[0] == "十":
            ten = "1"
            single = cn_to_num[i[1]]
            data["移轉層次"][con] = ten+single
            con += 1            
        else:
            ten = cn_to_num[i[0]]
            single = "0"
            data["移轉層次"][con] = ten+single            
            con += 1                   
    elif len(i) == 4:
        ten = cn_to_num[i[0]]
        single = cn_to_num[i[2]]
        data["移轉層次"][con] = ten+single            
        con += 1
    else:
        data["移轉層次"][con] = "0"
        con += 1
#從建物型態資料中篩選出公寓及透天厝 用於辨別該建物是否有電梯
p = "[公寓透天厝]{1,3}"
pattern = re.compile(p)
data_house_type = data["建物型態"]
elevator = []
for i in data_house_type:
    house_type = pattern.findall(i)
    if len(house_type) == 0:
        elevator.append("1")    
    else:
        elevator.append("0")
#將有電梯設成1 沒電梯設成0
data["電梯"] = np.array(elevator)
# data["電梯"] = elevator
#計算屋齡
house_age = (data["交易年月日"]-data["建築完成年月"])
house_age = house_age/10000
house_age = round(house_age)
data.insert(22,column="屋齡",value=house_age)
#將交易筆棟數裡的土地數、建物數、車位數分成三欄
amo = data["交易筆棟數"]
p2 = "\d+"
pattern2 = re.compile(p2)
land = []
house = []
park = []
for i in amo:
    amo2 = pattern2.findall(i)
    land.append(amo2[0])
    house.append(amo2[1])
    park.append(amo2[2])  
data.insert(6,column="車位數",value=park)
data.insert(6,column="建物數",value=house)
data.insert(6,column="土地數",value=land)
#自訂函數 將鄉鎮地區的中文轉成數字
def region(x):
    a = np.where(data["鄉鎮市區"] == x,1,0)
    data.insert(0,column=x,value=a)    
for i in set(data["鄉鎮市區"]):
    region(i)
#刪除不需要的欄位 以建立機器學習模型   
data2 = data.drop("鄉鎮市區",axis=1)
data2 = data2.drop("土地位置建物門牌",axis=1)
data2 = data2.drop("土地移轉總面積平方公尺",axis=1)
data2 = data2.drop("都市土地使用分區",axis=1)
data2 = data2.drop("交易年月日",axis=1)
data2 = data2.drop("建築完成年月",axis=1)
data2 = data2.drop("交易筆棟數",axis=1)
data2 = data2.drop("建物型態",axis=1)
data2 = data2.drop("總價元",axis=1)
data2 = data2.drop("單價元平方公尺",axis=1)
data2 = data2.drop("陽台面積",axis=1)
#設定機器學習的X，y
X = data2
y = data["總價元"]
#%%  長條折線圖  歷年平均單價，交易筆數
#輸入107-111年各年的實價登錄資料
#命名為"data+年份" 並計算出每坪平均價格
for i in range(107,112):
    globals()["data"+str(i)] = pd.read_csv(f"{i}.csv")
    globals()["avg_price"+str(i)] = globals()["data"+str(i)]["單價元平方公尺"]
    globals()["num"+str(i)] = len(globals()["avg_price"+str(i)])
    globals()["avg_price"+str(i)] = round(sum(globals()["avg_price"+str(i)])/len(globals()["avg_price"+str(i)])/10000/0.3025)
#繪圖
plt.figure(figsize=(21,9))
fig,f1 = plt.subplots()  
x1 = [i for i in range(107,112)]
y1 = [len(globals()["data"+str(i)]) for i in range(107,112)]
plt.ylim(0,10000)
plt.ylabel("交易筆數",size=20)
f1.bar(x1,y1,width=0.5,color="#D94600")
f2 =f1.twinx()
x2 = x1
y2 = [globals()["avg_price"+str(i)] for i in range(107,112) ]
plt.ylim(50,80)
plt.ylabel("每坪平均房價(萬元/坪)",size=20)
plt.title("歷年平均單價，交易筆數")
plt.grid(axis="y")
f2.plot(x2,y2,color="black")
plt.show()
#%% 110 各區平均房價
#讀取110房價資料
data110 = pd.read_csv("110.csv")
#自訂函數 將鄉鎮地區的中文轉成數字
def region110(x):
    a = np.where(data110["鄉鎮市區"] == x,1,0)
    data110.insert(0,column=x,value=a)    
for i in data110["鄉鎮市區"].unique():
    region110(i)
#計算各區交易總量    
data110_sum = data110.sum()
#建立自訂函數 用迴圈輪流取出各區的交易資料
region_num = []
region_price = []
def region_alone(x):
    data110_alone = data110
    data110_alone = data110_alone.sort_values(x)
    data110_alone = data110_alone.tail(data110_sum[x])
    return data110_alone 
for i in data110["鄉鎮市區"].unique():
    xregion_num = region_alone(i)    
    region_price.append(round(sum(xregion_num["單價元平方公尺"])/len(xregion_num)/10000/0.3025,2))
    region_num.append(data110_sum[i])
    
#繪圖
fig,f1 = plt.subplots(figsize=(15,9))
x1 = [i for i in range(12)]
y1 = [i for i in region_num]
f1.bar(x1,y1,width=0.5,color="#D94600")
plt.ylabel("交易筆數",size=20)
f2 =f1.twinx()
x2 = x1
y2 = region_price

f2.plot(x2,y2,color="black")
labels =list(data110["鄉鎮市區"].unique())
plt.xticks(x1,labels)

plt.ylabel("每坪平均房價(萬元/坪)",size=20)
plt.ylim(30,110)

plt.title("110年各區平均單價",size=30)
plt.grid(axis="y")
plt.savefig("110年各區平均單價")
plt.show()
#%%  地圖
#讀取全台地區經緯度資料 只能用絕對路徑
map_ = gpd.read_file(r"C:\Users\Administrator\Downloads\1030-20221031T012735Z-001\1030\mapdata202209220431\TOWN_MOI_1100415.shp")
map_ = map_.sort_values("COUNTYID")
# 從全台各地區經緯度資料中篩選出台北市
taipei = map_.head(12)
#取出各區每坪單價 並新增townid資料
avg_townid = pd.DataFrame(region_price)
avg_townid.columns = ["price"]
townid=["A15","A11","A01","A16","A02","A10","A05","A13","A17","A03","A14","A09"]
avg_townid.insert(1,column="TOWNID",value=townid)
#用townid當作媒介結合各區每坪單價跟經緯度兩個DataFrame
taipei_avgprice = taipei.merge(avg_townid)

#繪圖
fig, ax = plt.subplots(figsize=(12,16))
ax = taipei_avgprice.plot(ax=ax,column="price",cmap="OrRd",edgecolor='k',legend=True,legend_kwds={'shrink': 0.3})
region_name = ['北投區','內湖區','文山區','大安區','萬華區','中正區','南港區','松山區','信義區','大同區','中山區','士林區']
for i,j in zip(taipei_avgprice.representative_point(),region_name):
    ax.text(i.x,i.y,j,fontweight='bold',ha="center",va="center",size=12)
plt.title("各區平均單價",size=30)
ax.axis("off")
fig.savefig("各區平均單價地圖")
#%% rfr 建立隨機森林模型
rfr =RandomForestRegressor(n_estimators=100,max_depth=5,random_state=11)
r = 11
XTrain,XTest,yTrain,yTest=tts(X,y,test_size=0.3,random_state=r)

rfr.fit(XTrain,yTrain)
print(f"score = {rfr.score(XTest,yTest)}")
predy = rfr.predict(XTest)
print("MAPE =",np.mean(abs(yTest-predy)/yTest))
print("MAE =",metrics.mean_absolute_error(yTest,predy))
print("MSE =",metrics.mean_squared_error(yTest, predy))
print("RMSE =",np.sqrt(metrics.mean_squared_error(yTest,predy)))
#%% 熱力圖
data3 = data2
data3 = data3.insert(0,column="總價元",value=data["總價元"])
data3 = data3.drop(data3.iloc[:,1:13],axis=1)
plt.figure(figsize=(16,9),dpi=300)
df = data3.corr()
sns.heatmap(df,cmap="Reds",annot=True)
plt.show()
plt.savefig("熱力圖")

tmpdata = pd.get_dummies(data2)
#%% xgb 建立XGboost模型
XTrain = XTrain.astype(int)
yTrain = yTrain.astype(int)
XTest = XTest.astype(int)
yTest = yTest.astype(int)

data_matrix = xgb.DMatrix(XTrain,label=yTrain)
dtest = xgb.DMatrix(XTest,label=yTest)

xgb = xgb.XGBRegressor(objective="reg:linear",colsample_bytree=0.3
                       ,learning_rate=0.1,max_depth=5,alpha=10,n_estimators=1000)
xgb.fit(XTrain,yTrain)

print(xgb.feature_importances_)

testpred = xgb.predict(XTest)
print("R-squared =",xgb.score(XTest,yTest))
print("MAPE =",np.mean(abs(yTest-testpred)/yTest))
print("MAE =",metrics.mean_absolute_error(yTest,testpred))
print("MSE =",metrics.mean_squared_error(yTest, testpred))
print("RMSE =",np.sqrt(metrics.mean_squared_error(yTest,testpred)))
