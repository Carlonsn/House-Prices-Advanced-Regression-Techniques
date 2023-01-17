{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inizializzazione di PySpark e creazione della SparkSession\n",
    "\n",
    "import findspark\n",
    "\n",
    "location = findspark.find()\n",
    "findspark.init(location)\n",
    "\n",
    "import pyspark\n",
    "\n",
    "from pyspark.sql import SparkSession\n",
    "spark = SparkSession.builder.appName('house-prices').getOrCreate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "spark.sparkContext"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import delle librerie\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql.functions import *\n",
    "from pyspark.sql.types import StringType, IntegerType, DoubleType\n",
    "from pyspark.ml import Pipeline\n",
    "from pyspark.ml.feature import StringIndexer\n",
    "from pyspark.ml.feature import OneHotEncoder\n",
    "from pyspark.ml.feature import VectorAssembler\n",
    "from pyspark.ml.feature import StandardScaler\n",
    "from pyspark.ml.feature import PCA\n",
    "from pyspark.ml.regression import LinearRegression\n",
    "from pyspark.ml.tuning import ParamGridBuilder\n",
    "from pyspark.ml.tuning import CrossValidator\n",
    "from pyspark.ml.evaluation import RegressionEvaluator\n",
    "from pyspark.mllib.evaluation import RegressionMetrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lettura dei file csv training set e test set da HDFS\n",
    "\n",
    "trainDF = spark \\\n",
    "    .read \\\n",
    "    .csv('hdfs://localhost:9000/user/carlonsn/spark/input/train.csv',header = 'True', inferSchema='True', nullValue='NA')\n",
    "    \n",
    "testDF = spark \\\n",
    "    .read \\\n",
    "    .csv('hdfs://localhost:9000/user/carlonsn/spark/input/test.csv',header = 'True', inferSchema='True', nullValue='NA')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "display(trainDF)\n",
    "display(testDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"training set\")\n",
    "trainDF.printSchema()\n",
    "print(\"test set\")\n",
    "testDF.printSchema()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tot_row_train = trainDF.count()\n",
    "print('TRAINING SET\\nnumero di righe:', tot_row_train)\n",
    "n_features_train = len(trainDF.columns)\n",
    "print('numero features:', n_features_train)\n",
    "\n",
    "tot_row_test = testDF.count()\n",
    "print('TEST SET\\nnumero di righe:', tot_row_test)\n",
    "n_features_test = len(testDF.columns)\n",
    "print('numero features:', n_features_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualizzazione dei dati con Pandas\n",
    "pd_trainDF = trainDF.toPandas()\n",
    "pd_trainDF.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L'obiettivo del progetto è predire il prezzo delle case\n",
    "# Visualizziamo le statistiche di SalePrice \n",
    "\n",
    "trainDF.describe(['SalePrice']).show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualizziamo i prezzi delle case con un istogramma\n",
    "\n",
    "data = pd_trainDF['SalePrice']\n",
    "fig, ax = plt.subplots(constrained_layout=True)\n",
    "ax.hist(data, bins=150, linewidth=0.1, edgecolor=\"white\")\n",
    "fig.suptitle('Istogramma di SalePrice')\n",
    "ax.set_xlabel('SalePrice')\n",
    "ax.set_ylabel('Occorrenze')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calcolo la matrice di correlazione per evidenziare le features che hanno un'alta correlazione con SalePrice\n",
    "corr = pd_trainDF.drop('Id', axis=1).corr()\n",
    "corr.style.background_gradient(cmap='viridis', axis=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def my_plotter(ax, data1, xlabel, param_dict):\n",
    "   \n",
    "    out = ax.scatter(data1, 'S_P', data=data, **param_dict)\n",
    "    ax.set_xlabel(xlabel)\n",
    "    ax.set_ylabel('SalePrice')\n",
    "\n",
    "    return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {'S_P':  pd_trainDF['SalePrice'],\n",
    "        'O_Q':  pd_trainDF['OverallQual'],\n",
    "        'GLA':  pd_trainDF['GrLivArea'],\n",
    "        '1_F':  pd_trainDF['1stFlrSF'],\n",
    "        'TotB': pd_trainDF['TotalBsmtSF'],\n",
    "        'G_A':  pd_trainDF['GarageArea'],\n",
    "        'G_C':  pd_trainDF['GarageCars']\n",
    "        }\n",
    "\n",
    "\n",
    "fig, ([ax1,ax2],[ax3,ax4],[ax5,ax6]) = plt.subplots(3,2,figsize=(16, 10), constrained_layout=True)\n",
    "fig.suptitle('Features con alta correlazione')\n",
    "my_plotter(ax1, 'O_Q', 'Overall Quality', {'marker': 'x'})\n",
    "my_plotter(ax2, 'GLA', 'GrLivArea', {'marker': 'o', 'color': '#d45'})\n",
    "my_plotter(ax3, 'G_A', 'GarageArea', {'marker': 'o', 'color': '#073'})\n",
    "my_plotter(ax4, 'G_C', 'GarageCars', {'marker': 'x', 'color': '#073'})\n",
    "my_plotter(ax5, '1_F', '1stFlrSF', {'marker': 'o', 'color': '#aa2'})\n",
    "my_plotter(ax6, 'TotB', 'TotalBsmtSF', {'marker': 'o', 'color': '#ad2'});\n",
    "\n",
    "#le features con alta correlazione mostrano una dipendenza lineare con SalePrice "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verifica della presenza di valori nulli\n",
    "\n",
    "#TRAINING\n",
    "tot_null_val_train = pd_trainDF.isnull().sum()[pd_trainDF.isnull().sum()>0].sort_values(ascending=False)\n",
    "percent_null_val_train = tot_null_val_train / pd_trainDF.shape[0]*100\n",
    "\n",
    "#TEST\n",
    "pd_testDF = testDF.toPandas()\n",
    "tot_null_val_test = pd_testDF.isnull().sum()[pd_testDF.isnull().sum()>0].sort_values(ascending=False)\n",
    "percent_null_val_test = tot_null_val_test / pd_testDF.shape[0]*100\n",
    "\n",
    "missing = pd.concat([tot_null_val_train, percent_null_val_train ], axis=1,\\\n",
    "         keys=['Total_Train', 'Perc_missing_Train ( % )'])\n",
    "missing_test = pd.concat([tot_null_val_test, percent_null_val_test ], axis=1,\\\n",
    "         keys=['Total_Test', 'Perc_missing_Test ( % )'])\n",
    "                 \n",
    "missing\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visto il numero elevato dei valori nulli di alcune features \n",
    "# ho deciso di droppare le colonne con percentule di valori nulli\n",
    "# superiore a 80%\n",
    "\n",
    "trainDF = trainDF.drop('PoolQC', 'MiscFeature', 'Alley', 'Fence')\n",
    "testDF = testDF.drop('PoolQC', 'MiscFeature', 'Alley', 'Fence')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Gestione dei valori nulli\n",
    "# features numeriche, imputiamo con la media \n",
    "# features categoriche, imputiamo con il valore più frequente\n",
    "\n",
    "trainDF.describe(['LotFrontage']).show()\n",
    "testDF.describe(['LotFrontage']).show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# creiamo un unico dataframe per gestire i valori medi e più frequenti\n",
    "pivotDF = trainDF['LotFrontage','Fireplaces','FireplaceQu','MasVnrType','MasVnrArea'].\\\n",
    "            unionByName(testDF['LotFrontage','Fireplaces','FireplaceQu','MasVnrType','MasVnrArea'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "avg_Lf = pivotDF.select(avg(pivotDF.LotFrontage)).collect()[0][0]\n",
    "trainDF = trainDF.na.fill({'LotFrontage' : avg_Lf})\n",
    "testDF = testDF.na.fill({'LotFrontage' : avg_Lf})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pivotDF.where(col('Fireplaces')=='0').groupby('Fireplaces','FireplaceQu').count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# FireplaceQu assume valore nullo nelle case che non hanno fireplace\n",
    "# sostituisco con None (no fireplace)\n",
    "trainDF = trainDF.na.fill({'FireplaceQu' : 'None'})\n",
    "testDF = testDF.na.fill({'FireplaceQu' : 'None'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# le due colonne sono nulle insieme tranne in un record\n",
    "pivotDF.where(col('MasVnrType').isNull()).groupBy(['MasVnrType', 'MasVnrArea']).count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# analizziamo MasVnrType e prendiamo il valore più frequente diverso da None\n",
    "pivotDF.groupBy('MasVnrType').count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#per prima cosa vado ad individuare il valore nullo di MasVnrType con MasVnrArea uguale a 198\n",
    "#  nel test set e sostituisco il valore più frequente\n",
    "\n",
    "testDF = testDF.withColumn('MasVnrType',\\\n",
    "                when((col('MasVnrArea') == 198) & (col('MasVnrType').isNull()),\\\n",
    "                'BrkFace').otherwise(testDF['MasVnrType']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#per i restanti valori sostituisco None e 0 valori più frequenti\n",
    "trainDF = trainDF.na.fill({'MasVnrType' : 'None'})\n",
    "trainDF = trainDF.na.fill({'MasVnrArea' : 0})\n",
    "\n",
    "testDF = testDF.na.fill({'MasVnrType' : 'None'})\n",
    "testDF = testDF.na.fill({'MasVnrArea' : 0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainDF.groupBy('Electrical').count().show()\n",
    "testDF.groupBy('Utilities').count().show()\n",
    "testDF.groupBy('MSZoning').count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testDF.groupBy('SaleType').count().show()\n",
    "testDF.groupBy('KitchenQual').count().show()\n",
    "testDF.groupBy('Functional').count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sostituisco con il valore più frequente\n",
    "trainDF = trainDF.na.fill({'Electrical' : 'SBrkr'})\n",
    "testDF = testDF.na.fill({'Utilities' : 'AllPub'})\n",
    "testDF = testDF.na.fill({'MSZoning' : 'RL'})\n",
    "testDF = testDF.na.fill({'SaleType' : 'WD'})\n",
    "testDF = testDF.na.fill({'KitchenQual' : 'TA'})\n",
    "testDF = testDF.na.fill({'Functional' : 'Typ'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainDF.where(col('GarageYrBlt').isNull())\\\n",
    "    .groupBy(['GarageArea', 'GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']).count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GarageArea è 0, il garage non è presente\n",
    "trainDF = trainDF.na.fill({'GarageYrBlt' : 0})\n",
    "trainDF = trainDF.na.fill('None', subset=['GarageType','GarageFinish','GarageQual','GarageCond'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testDF.where(col('GarageYrBlt').isNull()).groupBy(['GarageCars','GarageArea', 'GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']).count().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# percentuale di case con 'YearBuilt' uguale a 'GarageYrBlt'\n",
    "egual_year = testDF.where(col('YearBuilt')==col('GarageYrBlt')).count()\n",
    "egual_year/tot_row_test*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testDF.where((col('GarageArea') == 360) & (col('GarageYrBlt').isNull())).select('YearBuilt').show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testDF = testDF.withColumn('GarageYrBlt',\\\n",
    "                when((col('GarageArea') == 360) & (col('GarageYrBlt').isNull()),\\\n",
    "                1910).otherwise(0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# i restanti per evitare una scorretta imputazione li pongo a 0 oppure None\n",
    "trainDF = trainDF.na.fill('None', subset=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'])\n",
    "testDF = testDF.na.fill(0, subset=[ 'TotalBsmtSF','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',  'BsmtFullBath',\n",
    "      'BsmtHalfBath', 'GarageCars', 'GarageArea','GarageYrBlt'])\n",
    "testDF = testDF.na.fill('None', subset=['Exterior1st', 'Exterior2nd', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',\n",
    "      'BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#====================================================================================#\n",
    "#                                       PIPELINE                                     #\n",
    "#====================================================================================#"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Seleziono dai dataset solo le features categoriche di tipo string\n",
    "\n",
    "string_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, StringType)]\n",
    "string_cols_test = [f.name for f in testDF.schema.fields if isinstance(f.dataType, StringType)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Unisco i dati in un unico dataframe \n",
    "\n",
    "labelDF = trainDF[string_cols].\\\n",
    "          unionByName(testDF[string_cols_test])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Indicizziamo i dati categorici su un unico dataframe\n",
    "\n",
    "indexers = [StringIndexer(inputCol=column, outputCol=column+\"_index\").fit(labelDF) for column in labelDF.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trasformiamo il dataset per ottenere le colonne indicizzate e utilizzarle come feature\n",
    "for indexer in indexers:\n",
    "    trainDF = indexer.transform(trainDF)\n",
    "    testDF = indexer.transform(testDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# le colonne indicizzate hanno tipo double\n",
    "\n",
    "int_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, IntegerType )]\n",
    "double_cols = [f.name for f in trainDF.schema.fields if isinstance(f.dataType, DoubleType )]\n",
    "\n",
    "double_cols_test = [f.name for f in testDF.schema.fields if isinstance(f.dataType, DoubleType )]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "corr = trainDF.select(*double_cols,*int_cols).toPandas().corr()\n",
    "corr[['SalePrice']][corr['SalePrice']>0].sort_values(by='SalePrice',ascending=False).style.background_gradient(cmap='viridis', axis=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# seleziono il numero di features che hanno valore di correlazione maggiore di 0.10\n",
    "# parametro da utilizzare nella PCA\n",
    "\n",
    "n_reduce_features = len(corr[['SalePrice']][corr['SalePrice']>0.10].axes[0].tolist())\n",
    "n_reduce_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoders = [OneHotEncoder()\\\n",
    ".setInputCol(column)\\\n",
    ".setOutputCol(column+'_encoded').fit(trainDF) for column in trainDF[double_cols].columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for encoder in encoders:\n",
    "    trainDF = encoder.transform(trainDF)\n",
    "    testDF = encoder.transform(testDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# seleziono le colonne numeriche e 'encoded' rimuovendo le colonne 'index' e di tipo string\n",
    "trainDF = trainDF.drop(*double_cols,*string_cols)\n",
    "testDF = testDF.drop(*double_cols_test,*string_cols_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testDF = testDF.withColumn('SalePrice', lit(0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "assembler = VectorAssembler(inputCols=trainDF.drop('SalePrice').columns,\n",
    "                           outputCol='features').setHandleInvalid(\"keep\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler(inputCol='features', outputCol='std_features', withStd=True, withMean=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pca = PCA().setInputCol('std_features').setOutputCol('pca_features').setK(n_reduce_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LinearRegression(featuresCol = 'pca_features', labelCol='SalePrice', tol=1e-6, standardization=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#lr.explainParams()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = Pipeline().setStages([assembler, scaler, pca, lr])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [100])\\\n",
    "                              .addGrid(lr.regParam, [0.1, 0.2, 0.8])\\\n",
    "                              .addGrid(lr.elasticNetParam, [0.1, 0.15, 0.2])\\\n",
    "                              .build()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_evaluator = RegressionEvaluator(predictionCol=\"prediction\", labelCol=\"SalePrice\", metricName=\"rmse\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv = CrossValidator().setEstimator(pipeline)\\\n",
    "                     .setEvaluator(lr_evaluator)\\\n",
    "                     .setEstimatorParamMaps(paramGrid)\\\n",
    "                     .setNumFolds(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Addestriamo sul training set\n",
    "cvModel = cv.fit(trainDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# utilizziamo il modello addestrato per fare la predizione sul test set\n",
    "predictions_test = cvModel.transform(testDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "performance_train = cvModel.transform(trainDF)\n",
    "print(\"RMSE sul training set = %g\" % lr_evaluator.evaluate(performance_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "output_train = performance_train.select(log( performance_train.prediction), log(performance_train.SalePrice))\\\n",
    "                                .rdd.map( lambda x: (float(x[0]), float(x[1])));\n",
    "metrics_train = RegressionMetrics(output_train);\n",
    "\n",
    "print(\"RMSE dei log sul training set = %g\" %metrics_train.rootMeanSquaredError)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_test.select('id','prediction').show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Salviamo il modello addestrato\n",
    "\n",
    "cvModel.write().overwrite().save('hdfs://localhost:9000/user/carlonsn/spark/output/Linear_Regressor_cv_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Salviamo le colonne \"Id\" \"SalePrice\" in un file csv\n",
    "\n",
    "predictions_test\\\n",
    "  .withColumn(\"SalePrice\", col(\"prediction\"))\\\n",
    "  .select(\"Id\", \"SalePrice\")\\\n",
    "  .coalesce(1)\\\n",
    "  .write\\\n",
    "  .csv('hdfs://localhost:9000/user/carlonsn/spark/output/house_prices_predictions.csv',\\\n",
    "    header=True, mode='overwrite')"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "f8e42924433a441d517fb53969ed8937beff2c25942fc7b2efda8cfeb42c019b"
  },
  "kernelspec": {
   "display_name": "Python 3.9.13 64-bit (windows store)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
