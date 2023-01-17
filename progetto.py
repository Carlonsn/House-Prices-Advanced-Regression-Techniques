# Inizializzazione di PySpark e creazione della SparkSession

import findspark

location = findspark.find()
findspark.init(location)

import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('house-prices').getOrCreate()
spark.sparkContext

# Import delle librerie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

# Lettura dei file csv training set e test set da HDFS

trainDF = spark \
    .read \
    .csv('hdfs://localhost:9000/user/carlonsn/spark/input/train.csv',header = 'True', inferSchema='True', nullValue='NA')
    
testDF = spark \
    .read \
    .csv('hdfs://localhost:9000/user/carlonsn/spark/input/test.csv',header = 'True', inferSchema='True', nullValue='NA')  

display(trainDF)
display(testDF)

print("training set")
trainDF.printSchema()

print("test set")
testDF.printSchema()

tot_row_train = trainDF.count()
print('TRAINING SET\nnumero di righe:', tot_row_train)

n_features_train = len(trainDF.columns)
print('numero features:', n_features_train)

tot_row_test = testDF.count()
print('TEST SET\nnumero di righe:', tot_row_test)

n_features_test = len(testDF.columns)
print('numero features:', n_features_test)

# Visualizzazione dei dati con Pandas
pd_trainDF = trainDF.toPandas()
pd_trainDF.head()

# L'obiettivo del progetto è predire il prezzo delle case
# Visualizziamo le statistiche di SalePrice 

trainDF.describe(['SalePrice']).show()

# Visualizziamo i prezzi delle case con un istogramma

data = pd_trainDF['SalePrice']
fig, ax = plt.subplots(constrained_layout=True)
ax.hist(data, bins=150, linewidth=0.1, edgecolor="white")
fig.suptitle('Istogramma di SalePrice')
ax.set_xlabel('SalePrice')
ax.set_ylabel('Occorrenze')
plt.show()

# Calcolo la matrice di correlazione per evidenziare le features che hanno un'alta correlazione con SalePrice
corr = pd_trainDF.drop('Id', axis=1).corr()
corr.style.background_gradient(cmap='viridis', axis=None)

def my_plotter(ax, data1, xlabel, param_dict):
   
    out = ax.scatter(data1, 'S_P', data=data, **param_dict)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('SalePrice')

    return out
data = {'S_P':  pd_trainDF['SalePrice'],
        'O_Q':  pd_trainDF['OverallQual'],
        'GLA':  pd_trainDF['GrLivArea'],
        '1_F':  pd_trainDF['1stFlrSF'],
        'TotB': pd_trainDF['TotalBsmtSF'],
        'G_A':  pd_trainDF['GarageArea'],
        'G_C':  pd_trainDF['GarageCars']
        }


fig, ([ax1,ax2],[ax3,ax4],[ax5,ax6]) = plt.subplots(3,2,figsize=(16, 10), constrained_layout=True)
fig.suptitle('Features con alta correlazione')
my_plotter(ax1, 'O_Q', 'Overall Quality', {'marker': 'x'})
my_plotter(ax2, 'GLA', 'GrLivArea', {'marker': 'o', 'color': '#d45'})
my_plotter(ax3, 'G_A', 'GarageArea', {'marker': 'o', 'color': '#073'})
my_plotter(ax4, 'G_C', 'GarageCars', {'marker': 'x', 'color': '#073'})
my_plotter(ax5, '1_F', '1stFlrSF', {'marker': 'o', 'color': '#aa2'})
my_plotter(ax6, 'TotB', 'TotalBsmtSF', {'marker': 'o', 'color': '#ad2'});

#le features con alta correlazione mostrano una dipendenza lineare con SalePrice 
# Verifica della presenza di valori nulli

#TRAINING
tot_null_val_train = pd_trainDF.isnull().sum()[pd_trainDF.isnull().sum()>0].sort_values(ascending=False)
percent_null_val_train = tot_null_val_train / pd_trainDF.shape[0]*100

#TEST
pd_testDF = testDF.toPandas()
tot_null_val_test = pd_testDF.isnull().sum()[pd_testDF.isnull().sum()>0].sort_values(ascending=False)
percent_null_val_test = tot_null_val_test / pd_testDF.shape[0]*100

missing = pd.concat([tot_null_val_train, percent_null_val_train ], axis=1,\
         keys=['Total_Train', 'Perc_missing_Train ( % )'])
missing_test = pd.concat([tot_null_val_test, percent_null_val_test ], axis=1,\
         keys=['Total_Test', 'Perc_missing_Test ( % )'])
                 
missing
missing_test

# Visto il numero elevato dei valori nulli di alcune features 
# ho deciso di droppare le colonne con percentule di valori nulli
# superiore a 80%

trainDF = trainDF.drop('PoolQC', 'MiscFeature', 'Alley', 'Fence')
testDF = testDF.drop('PoolQC', 'MiscFeature', 'Alley', 'Fence')

# Gestione dei valori nulli
# features numeriche, imputiamo con la media 
# features categoriche, imputiamo con il valore più frequente

trainDF.describe(['LotFrontage']).show()
testDF.describe(['LotFrontage']).show()

# creiamo un unico dataframe per gestire i valori medi e più frequenti

pivotDF = trainDF['LotFrontage','Fireplaces','FireplaceQu','MasVnrType','MasVnrArea'].\
            unionByName(testDF['LotFrontage','Fireplaces','FireplaceQu','MasVnrType','MasVnrArea'])
avg_Lf = pivotDF.select(avg(pivotDF.LotFrontage)).collect()[0][0]
trainDF = trainDF.na.fill({'LotFrontage' : avg_Lf})
testDF = testDF.na.fill({'LotFrontage' : avg_Lf})
pivotDF.where(col('Fireplaces')=='0').groupby('Fireplaces','FireplaceQu').count().show()

# FireplaceQu assume valore nullo nelle case che non hanno fireplace
# sostituisco con None (no fireplace)

trainDF = trainDF.na.fill({'FireplaceQu' : 'None'})
testDF = testDF.na.fill({'FireplaceQu' : 'None'})

# le due colonne sono nulle insieme tranne in un record

pivotDF.where(col('MasVnrType').isNull()).groupBy(['MasVnrType', 'MasVnrArea']).count().show()

# analizziamo MasVnrType e prendiamo il valore più frequente diverso da None

pivotDF.groupBy('MasVnrType').count().show()

#per prima cosa vado ad individuare il valore nullo di MasVnrType con MasVnrArea uguale a 198
#  nel test set e sostituisco il valore più frequente

testDF = testDF.withColumn('MasVnrType',\
                when((col('MasVnrArea') == 198) & (col('MasVnrType').isNull()),\
                'BrkFace').otherwise(testDF['MasVnrType']))

#per i restanti valori sostituisco None e 0 valori più frequenti

trainDF = trainDF.na.fill({'MasVnrType' : 'None'})
trainDF = trainDF.na.fill({'MasVnrArea' : 0})

testDF = testDF.na.fill({'MasVnrType' : 'None'})
testDF = testDF.na.fill({'MasVnrArea' : 0})
trainDF.groupBy('Electrical').count().show()
testDF.groupBy('Utilities').count().show()
testDF.groupBy('MSZoning').count().show()
testDF.groupBy('SaleType').count().show()
testDF.groupBy('KitchenQual').count().show()
testDF.groupBy('Functional').count().show()

#sostituisco con il valore più frequente

trainDF = trainDF.na.fill({'Electrical' : 'SBrkr'})
testDF = testDF.na.fill({'Utilities' : 'AllPub'})
testDF = testDF.na.fill({'MSZoning' : 'RL'})
testDF = testDF.na.fill({'SaleType' : 'WD'})
testDF = testDF.na.fill({'KitchenQual' : 'TA'})
testDF = testDF.na.fill({'Functional' : 'Typ'})
trainDF.where(col('GarageYrBlt').isNull())\
    .groupBy(['GarageArea', 'GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']).count().show()

# GarageArea è 0, il garage non è presente

trainDF = trainDF.na.fill({'GarageYrBlt' : 0})
trainDF = trainDF.na.fill('None', subset=['GarageType','GarageFinish','GarageQual','GarageCond'])
testDF.where(col('GarageYrBlt').isNull()).groupBy(['GarageCars','GarageArea', 'GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']).count().show()

# percentuale di case con 'YearBuilt' uguale a 'GarageYrBlt'

egual_year = testDF.where(col('YearBuilt')==col('GarageYrBlt')).count()
egual_year/tot_row_test*100
testDF.where((col('GarageArea') == 360) & (col('GarageYrBlt').isNull())).select('YearBuilt').show()
testDF = testDF.withColumn('GarageYrBlt',\
                when((col('GarageArea') == 360) & (col('GarageYrBlt').isNull()),\
                1910).otherwise(0))

# i restanti per evitare una scorretta imputazione li pongo a 0 oppure None

trainDF = trainDF.na.fill('None', subset=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'])
testDF = testDF.na.fill(0, subset=[ 'TotalBsmtSF','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',  'BsmtFullBath',
      'BsmtHalfBath', 'GarageCars', 'GarageArea','GarageYrBlt'])
testDF = testDF.na.fill('None', subset=['Exterior1st', 'Exterior2nd', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
      'BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond'])

#====================================================================================#
#                                       PIPELINE                                     #
#====================================================================================#
#Seleziono dai dataset solo le features categoriche di tipo string

string_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, StringType)]
string_cols_test = [f.name for f in testDF.schema.fields if isinstance(f.dataType, StringType)]

# Unisco i dati in un unico dataframe 

labelDF = trainDF[string_cols].\
          unionByName(testDF[string_cols_test])

# Indicizziamo i dati categorici su un unico dataframe

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(labelDF) for column in labelDF.columns]

# Trasformiamo il dataset per ottenere le colonne indicizzate e utilizzarle come feature
for indexer in indexers:
    trainDF = indexer.transform(trainDF)
    testDF = indexer.transform(testDF)

# le colonne indicizzate hanno tipo double

int_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, IntegerType )]
double_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, DoubleType )]

double_cols_test = [f.name for f in testDF.schema.fields if isinstance(f.dataType, DoubleType )]
corr = trainDF.select(*double_cols,*int_cols).toPandas().corr()
corr[['SalePrice']][corr['SalePrice']>0].sort_values(by='SalePrice',ascending=False).style.background_gradient(cmap='viridis', axis=None)

# seleziono il numero di features che hanno valore di correlazione maggiore di 0.10
# parametro da utilizzare nella PCA

n_reduce_features = len(corr[['SalePrice']][corr['SalePrice']>0.10].axes[0].tolist())
n_reduce_features
encoders = [OneHotEncoder()\
.setInputCol(column)\
.setOutputCol(column+'_encoded').fit(trainDF) for column in trainDF[double_cols].columns]
for encoder in encoders:
    trainDF = encoder.transform(trainDF)
    testDF = encoder.transform(testDF)

# seleziono le colonne numeriche e 'encoded' rimuovendo le colonne 'index' e di tipo string

trainDF = trainDF.drop(*double_cols,*string_cols)
testDF = testDF.drop(*double_cols_test,*string_cols_test)
testDF = testDF.withColumn('SalePrice', lit(0))
assembler = VectorAssembler(inputCols=trainDF.drop('SalePrice').columns,
                           outputCol='features').setHandleInvalid("keep")
scaler = StandardScaler(inputCol='features', outputCol='std_features', withStd=True, withMean=True)
pca = PCA().setInputCol('std_features').setOutputCol('pca_features').setK(n_reduce_features)
lr = LinearRegression(featuresCol = 'pca_features', labelCol='SalePrice', tol=1e-6, standardization=False)

#lr.explainParams()
pipeline = Pipeline().setStages([assembler, scaler, pca, lr])
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [100])\
                              .addGrid(lr.regParam, [0.1, 0.2, 0.8])\
                              .addGrid(lr.elasticNetParam, [0.1, 0.15, 0.2])\
                              .build()

lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="rmse")

cv = CrossValidator().setEstimator(pipeline)\
                     .setEvaluator(lr_evaluator)\
                     .setEstimatorParamMaps(paramGrid)\
                     .setNumFolds(10)

# Addestriamo sul training set

cvModel = cv.fit(trainDF)

# utilizziamo il modello addestrato per fare la predizione sul test set

predictions_test = cvModel.transform(testDF)
performance_train = cvModel.transform(trainDF)
print("RMSE sul training set = %g" % lr_evaluator.evaluate(performance_train))
output_train = performance_train.select(log( performance_train.prediction), log(performance_train.SalePrice))\
                                .rdd.map( lambda x: (float(x[0]), float(x[1])));
metrics_train = RegressionMetrics(output_train);

print("RMSE dei log sul training set = %g" %metrics_train.rootMeanSquaredError)
predictions_test.select('id','prediction').show()

# Salviamo il modello addestrato

cvModel.write().overwrite().save('hdfs://localhost:9000/user/carlonsn/spark/output/Linear_Regressor_cv_model')

# Salviamo le colonne "Id" "SalePrice" in un file csv

predictions_test\
  .withColumn("SalePrice", col("prediction"))\
  .select("Id", "SalePrice")\
  .coalesce(1)\
  .write\
  .csv('hdfs://localhost:9000/user/carlonsn/spark/output/house_prices_predictions.csv',\
    header=True, mode='overwrite')