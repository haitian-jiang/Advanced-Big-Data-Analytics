# set environment variables for runtime
import os
os.environ["JAVA_HOME"]="/opt/homebrew/Cellar/openjdk@11/11.0.12/"
os.environ["PYSPARK_PYTHON"]="/opt/miniconda3/envs/spark/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"]="/opt/miniconda3/envs/spark/bin/python"

from pyspark import SparkContext, SparkConf
from operator import add, itemgetter
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

conf = SparkConf().setAppName("jhtnb").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

data_path = '../mernis/data_dump.sql'
dataset = sc.textFile(data_path)
dataset = dataset.map(lambda x: x.split('\t'))
# drop unneeded lines
dataset = dataset.filter(lambda x: x[0] != "uid")

# frequently used functions
from operator import add  # a more elegant way
is_male = lambda row: row[6] == "E"
is_female = lambda row: row[6] == "K"

# add last column as age
dataset = dataset.map(lambda x:x+[2009-int(x[8].split('/')[2])+1])

eldest_man = dataset.filter(is_male)\
            .sortBy(lambda x: int((x[8].split('/'))[2]))\
            .filter(lambda x: x[17] < 150).take(1)

print(f"The eldest man is {eldest_man[0][2]} {eldest_man[0][3]} if alive.")

# turn each row into a list of (char, 1)
char_ize = lambda r: list(map(lambda c:(c,1), r['first_name']+r['last_name']))
most_freq = dataset.flatMap(char_ize).reduceByKey(add).sortBy(lambda x:-x[1]).take(1)
print(f"The most frequent char is {most_freq[0][0]}, with frequency {most_freq[0][1]}.")

def age_bin(row):
    upper_bound = [18, 28, 38 ,48, 59, float("inf")]
    for i, bnd in enumerate(upper_bound):
        if row[17] <= bnd:
            return (i, 1)

male = dataset.filter(is_male).count()
female = dataset.filter(is_female).count()
print(f"Male: {male}; female: {female}")

male_month = dataset.filter(is_male)\
             .map(lambda x: ((x[8].split("/"))[1], 1))\
             .reduceByKey(add).collect()

female_month = dataset.filter(is_female)\
               .map(lambda x: ((x[8].split("/"))[1], 1))\
               .reduceByKey(add).collect()

street = dataset.map(lambda x:(f"{x[11]}/{x[12]}/{x[13]}/{x[14]}", 1)).reduceByKey(add).collect()

street.sort(key=lambda x:-x[1])

plus_60 = dataset.filter(lambda x:x[17]>=60).map(lambda x:(x[11],1)).reduceByKey(add).sortBy(lambda x:-x[1]).collect()

all_pop = dataset.map(lambda x:(x[11],1)).reduceByKey(add).collect()

d = dict(plus_60)
old_prop = [(i[0], d[i[0]]/i[1]) for i in all_pop]
old_prop.sort(key=lambda x:-x[1])
old_prop[:10]

male_last = dataset.filter(is_male)\
            .map(lambda x: (x[3], 1)).reduceByKey(add)\
            .sortBy(lambda x:-x[1]).collect()
female_last = dataset.filter(is_female)\
              .map(lambda x: (x[3], 1)).reduceByKey(add)\
              .sortBy(lambda x:-x[1]).collect()

# use address city
male_city = dataset.filter(is_male).map(lambda x:(x[11],1)).reduceByKey(add).collect()
female_city = dataset.filter(is_female).map(lambda x:(x[11],1)).reduceByKey(add).collect()

male_city_d = dict(male_city)
female_city_d = dict(female_city)
sex_ratio = [(city, male_city_d[city]/female_city_d[city]) for city in male_city_d.keys()]

sex_ratio.sort(key=lambda x:x[1])
print(sex_ratio[:5])
print(sex_ratio[-5:])

dataset.map(lambda row: (row[8],1)).reduceByKey(add).sortBy(lambda x:-x[1]).take(10)

dob = dataset.map(lambda x: (x[8],1)).reduceByKey(add).sortBy(lambda x:-x[1]).collect()

total_age = dataset.map(lambda x: (x[11], x[17])).reduceByKey(add).collect()
population = dataset.map(lambda x:(x[11], 1)).reduceByKey(add).collect()

age = dict(total_age)
pop = dict(population)
for city in age:
    age[city] /= pop[city]
avg_age = list(age.items())
avg_age.sort()
avg_age[:10]

age_60 = dataset.filter(lambda x:x[17]>=60).map(lambda x:(x[11],1)).reduceByKey(add).collect()
age_65 = dataset.filter(lambda x:x[17]>=65).map(lambda x:(x[11],1)).reduceByKey(add).collect()

over_60 = dict(age_60)
over_65 = dict(age_65)
for city in pop:
    over_60[city] /= pop[city]
    over_65[city] /= pop[city]
aging = {city: (over_60[city], over_65[city], over_60[city]>0.1 or over_65[city]>0.07) for city in pop}
list(aging.items())[:10]

population.sort(key=lambda x:-x[1])
ten_largest = [i[0] for i in population[:10]]

birth_month = dataset.filter(lambda x:x[11] in ten_largest).map(lambda x: ((x[11], x[8].split("/")[1]),1)).reduceByKey(add).collect()

from collections import defaultdict
city_month = defaultdict(list)
for m in birth_month:
    city_month[m[0][0]].append((m[0][1], m[1]))
for c in city_month:
    city_month[c].sort(key=lambda x:-x[1])
    city_month[c] = city_month[c][:2]

last_name = dataset.filter(lambda x:x[11] in ten_largest).map(lambda x: ((x[11], x[3]),1)).reduceByKey(add).collect()

from collections import defaultdict
city_lname = defaultdict(list)
for m in last_name:
    city_lname[m[0][0]].append((m[0][1], m[1]))
for c in city_lname:
    city_lname[c].sort(key=lambda x:-x[1])
    city_lname[c] = city_lname[c][:3]

city_area = {'ISTANBUL':5343, 'KONYA':38873, 'IZMIR':11891, \
             'ANKARA':24521, 'BURSA':1036,'SIVAS':2768, 'SAMSUN':1055, \
             'AYDIN':1582, 'ADANA':1945, 'SANLIURFA':18584}

for c in city_area:
    print(c, f"\t{pop[c]/city_area[c]:.2f}")

national_pop = dataset.count()
city_mig = dataset.filter(lambda x: x[9]!=x[11]).count()
district_mig = dataset.filter(lambda x: x[10]!=x[12]).count()

# load data
from pyspark.sql import SparkSession
spark = SparkSession.builder.config("spark.driver.memory", "15g").getOrCreate()
dataframe = spark.read.format('csv').option('sep','\t').option('header','true').load(data_path)

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# add label column, the name "label" is by default
label_indexer = StringIndexer(inputCol="address_city", outputCol="label")
dataframe = label_indexer.fit(dataframe).transform(dataframe)
# use district and neighborhood as feature
district_indexer = StringIndexer(inputCol="address_district", outputCol="district_feature")
dataframe = district_indexer.fit(dataframe).transform(dataframe)
neighbor_indexer = StringIndexer(inputCol="address_neighborhood", outputCol="neighbor_feature")
dataframe = neighbor_indexer.fit(dataframe).transform(dataframe)

encoder = OneHotEncoder(inputCols=["district_feature","neighbor_feature"],outputCols=["district_vec","neighbor_vec"])
ohe = encoder.fit(dataframe).transform(dataframe)
# the name "features" is by default
assembler = VectorAssembler(inputCols=["district_vec", "neighbor_vec"],outputCol="features")
df_h1 = assembler.transform(ohe)

train, valid, test = df_h1.randomSplit([0.7, 0.1, 0.2])
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model_h1 = nb.fit(train)

y_pred_val = model_h1.transform(valid)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc_val = evaluator.evaluate(y_pred_val)
acc_val

y_pred_test = model_h1.transform(test)
y_prop_test = y_pred_test.select("probability","label").rdd

def top_K_acc(rows, K):
    in_top_K = lambda row: int(row[1]) in \
               [i[1] for i in 
                sorted(list(zip(row[0], range(len(row[0])))), reverse=True)[:K]]
    hit = rows.filter(in_top_K).count()
    tot = rows.count()
    return hit / tot

for K in range(1,6):
    acc = top_K_acc(y_prop_test, K)
    print(f"Top {K} accuracy is {acc}")

# add label column
gender_indexer = StringIndexer(inputCol="gender", outputCol="label")
dataframe = gender_indexer.fit(dataframe).transform(dataframe)
# use name as feature
name_indexer = StringIndexer(inputCol="first", outputCol="name_feature")
dataframe = name_indexer.fit(dataframe).transform(dataframe)

encoder = OneHotEncoder(inputCol="name_feature",outputCol="name_vec")
ohe = encoder.fit(dataframe).transform(dataframe)
assembler = VectorAssembler(inputCols=["name_vec"],outputCol="features")
df_h2 = assembler.transform(ohe)

train, valid, test = df_h2.randomSplit([0.7, 0.1, 0.2])
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model_h2 = nb.fit(train)

y_pred_val = model_h2.transform(valid)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc_val = evaluator.evaluate(y_pred_val)
acc_val

y_pred_test = model_h2.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc_test = evaluator.evaluate(y_pred_test)
acc_test

train, valid, test = dataset.randomSplit([0.7, 0.1, 0.2])

freq_last = train.map(lambda x: (x[3], 1)).reduceByKey(add)\
            .sortBy(lambda x:-x[1]).take(5)
freq_last = [i[0] for i in freq_last]

population = test.count()
for K in range(1, 6):
    hit = test.filter(lambda x:x[3] in freq_last[:K]).count()
    print(f"Top {K} accuracy is {hit/population}")

from pyspark.sql import SQLContext, Row
year_cnt = dataset.map(lambda x: (int((x[8].split("/"))[2]), 1)).reduceByKey(add).sortBy(lambda x:x[0]).filter(lambda x:x[0]>=1920)\
.map(lambda row: Row(features=Vectors.dense(row[0]), label=row[1])).toDF()

train, valid, test = year_cnt.randomSplit([0.7, 0.1, 0.2])

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

model_h4 = lr.fit(train)

valid_pred = model_h4.transform(valid)
valid_pred.show()

test_pred = model_h4.transform(test)
test_pred.show()

from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

model_h4_glr = glr.fit(train)

valid_pred_glr = model_h4.transform(valid)
valid_pred_glr.show()

city_diff = dataset.filter(lambda row: (row[9]=="MALATYA" and row[10]=="KULUNCAK")^(row[11]=="MALATYA" and row[12]=="KULUNCAK"))\
            .map(lambda row: (int(row[8].split("/")[2]), 1)).reduceByKey(add)

from pyspark.sql import SQLContext, Row
mig_year_cnt = city_diff.sortBy(lambda x:x[0]).filter(lambda x:x[0]>=1920)\
.map(lambda row: Row(features=Vectors.dense(row[0]), label=row[1])).toDF()

train, valid, test = mig_year_cnt.randomSplit([0.7, 0.1, 0.2])

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

model_h5 = lr.fit(train)

valid_pred_h5 = model_h5.transform(valid)
valid_pred_h5.show()

test_pred_h5 = model_h5.transform(test)
test_pred_h5.show()

