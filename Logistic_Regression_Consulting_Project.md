# Logistic Regression Consulting Project

## Binary Customer Churn

A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients. They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager. Luckily they have some historical data, can you help them out? Create a classification algorithm that will help classify whether or not a customer churned. Then the company can test this against incoming data for future customers to predict which customers will churn and assign them an account manager.

The data is saved as customer_churn.csv. Here are the fields and their definitions:

    Name : Name of the latest contact at Company
    Age: Customer Age
    Total_Purchase: Total Ads Purchased
    Account_Manager: Binary 0=No manager, 1= Account manager assigned
    Years: Totaly Years as a customer
    Num_sites: Number of websites that use the service.
    Onboard_date: Date that the name of the latest contact was onboarded
    Location: Client HQ Address
    Company: Name of Client Company
    
Once you've created the model and evaluated it, test out the model on some new data (you can think of this almost like a hold-out set) that your client has provided, saved under new_customers.csv. The client wants to know which customers are most likely to churn given this data (they don't have the label yet).


```python
import findspark

findspark.init('/home/brouxlis/spark-3.4.0-bin-hadoop3')

import pyspark
```


```python
from pyspark.sql import SparkSession
```


```python
spark = SparkSession.builder.appName('churn').getOrCreate()
```

    23/06/01 15:47:05 WARN Utils: Your hostname, brouxlis-VirtualBox resolves to a loopback address: 127.0.1.1; using 10.0.2.15 instead (on interface enp0s3)
    23/06/01 15:47:05 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    23/06/01 15:47:06 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    23/06/01 15:47:09 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
    23/06/01 15:47:09 WARN Utils: Service 'SparkUI' could not bind on port 4041. Attempting port 4042.



```python
df = spark.read.csv('customer_churn.csv', inferSchema=True, header=True)
```

                                                                                    


```python
df.show()
```

    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+
    |              Names| Age|Total_Purchase|Account_Manager|Years|Num_Sites|       Onboard_date|            Location|             Company|Churn|
    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+
    |   Cameron Williams|42.0|       11066.8|              0| 7.22|      8.0|2013-08-30 07:00:40|10265 Elizabeth M...|          Harvey LLC|    1|
    |      Kevin Mueller|41.0|      11916.22|              0|  6.5|     11.0|2013-08-13 00:38:46|6157 Frank Garden...|          Wilson PLC|    1|
    |        Eric Lozano|38.0|      12884.75|              0| 6.67|     12.0|2016-06-29 06:20:07|1331 Keith Court ...|Miller, Johnson a...|    1|
    |      Phillip White|42.0|       8010.76|              0| 6.71|     10.0|2014-04-22 12:43:12|13120 Daniel Moun...|           Smith Inc|    1|
    |     Cynthia Norton|37.0|       9191.58|              0| 5.56|      9.0|2016-01-19 15:31:15|765 Tricia Row Ka...|          Love-Jones|    1|
    |   Jessica Williams|48.0|      10356.02|              0| 5.12|      8.0|2009-03-03 23:13:37|6187 Olson Mounta...|        Kelly-Warren|    1|
    |        Eric Butler|44.0|      11331.58|              1| 5.23|     11.0|2016-12-05 03:35:43|4846 Savannah Roa...|   Reynolds-Sheppard|    1|
    |      Zachary Walsh|32.0|       9885.12|              1| 6.92|      9.0|2006-03-09 14:50:20|25271 Roy Express...|          Singh-Cole|    1|
    |        Ashlee Carr|43.0|       14062.6|              1| 5.46|     11.0|2011-09-29 05:47:23|3725 Caroline Str...|           Lopez PLC|    1|
    |     Jennifer Lynch|40.0|       8066.94|              1| 7.11|     11.0|2006-03-28 15:42:45|363 Sandra Lodge ...|       Reed-Martinez|    1|
    |       Paula Harris|30.0|      11575.37|              1| 5.22|      8.0|2016-11-13 13:13:01|Unit 8120 Box 916...|Briggs, Lamb and ...|    1|
    |     Bruce Phillips|45.0|       8771.02|              1| 6.64|     11.0|2015-05-28 12:14:03|Unit 1895 Box 094...|    Figueroa-Maynard|    1|
    |       Craig Garner|45.0|       8988.67|              1| 4.84|     11.0|2011-02-16 08:10:47|897 Kelley Overpa...|     Abbott-Thompson|    1|
    |       Nicole Olson|40.0|       8283.32|              1|  5.1|     13.0|2012-11-22 05:35:03|11488 Weaver Cape...|Smith, Kim and Ma...|    1|
    |     Harold Griffin|41.0|       6569.87|              1|  4.3|     11.0|2015-03-28 02:13:44|1774 Peter Row Ap...|Snyder, Lee and M...|    1|
    |       James Wright|38.0|      10494.82|              1| 6.81|     12.0|2015-07-22 08:38:40|45408 David Path ...|      Sanders-Pierce|    1|
    |      Doris Wilkins|45.0|       8213.41|              1| 7.35|     11.0|2006-09-03 06:13:55|28216 Wright Moun...|Andrews, Adams an...|    1|
    |Katherine Carpenter|43.0|      11226.88|              0| 8.08|     12.0|2006-10-22 04:42:38|Unit 4948 Box 481...|Morgan, Phillips ...|    1|
    |     Lindsay Martin|53.0|       5515.09|              0| 6.85|      8.0|2015-10-07 00:27:10|69203 Crosby Divi...|      Villanueva LLC|    1|
    |        Kathy Curry|46.0|        8046.4|              1| 5.69|      8.0|2014-11-06 23:47:14|9569 Caldwell Cre...|Berry, Orr and Ca...|    1|
    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+
    only showing top 20 rows
    



```python
df.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Churn: integer (nullable = true)
    



```python
df.describe().show()
```

    23/06/01 15:49:20 WARN package: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
    [Stage 7:>                                                          (0 + 1) / 1]

    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |summary|        Names|              Age|   Total_Purchase|   Account_Manager|            Years|         Num_Sites|            Location|             Company|              Churn|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |  count|          900|              900|              900|               900|              900|               900|                 900|                 900|                900|
    |   mean|         null|41.81666666666667|10062.82403333334|0.4811111111111111| 5.27315555555555| 8.587777777777777|                null|                null|0.16666666666666666|
    | stddev|         null|6.127560416916251|2408.644531858096|0.4999208935073339|1.274449013194616|1.7648355920350969|                null|                null| 0.3728852122772358|
    |    min|   Aaron King|             22.0|            100.0|                 0|              1.0|               3.0|00103 Jeffrey Cre...|     Abbott-Thompson|                  0|
    |    max|Zachary Walsh|             65.0|         18026.01|                 1|             9.15|              14.0|Unit 9800 Box 287...|Zuniga, Clark and...|                  1|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    


                                                                                    


```python
df.columns
```




    ['Names',
     'Age',
     'Total_Purchase',
     'Account_Manager',
     'Years',
     'Num_Sites',
     'Onboard_date',
     'Location',
     'Company',
     'Churn']




```python
from pyspark.sql.functions import year
```


```python
df = df.withColumn('Onboard_Year', year(df['Onboard_date']))
df.show()
```

    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+------------+
    |              Names| Age|Total_Purchase|Account_Manager|Years|Num_Sites|       Onboard_date|            Location|             Company|Churn|Onboard_Year|
    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+------------+
    |   Cameron Williams|42.0|       11066.8|              0| 7.22|      8.0|2013-08-30 07:00:40|10265 Elizabeth M...|          Harvey LLC|    1|        2013|
    |      Kevin Mueller|41.0|      11916.22|              0|  6.5|     11.0|2013-08-13 00:38:46|6157 Frank Garden...|          Wilson PLC|    1|        2013|
    |        Eric Lozano|38.0|      12884.75|              0| 6.67|     12.0|2016-06-29 06:20:07|1331 Keith Court ...|Miller, Johnson a...|    1|        2016|
    |      Phillip White|42.0|       8010.76|              0| 6.71|     10.0|2014-04-22 12:43:12|13120 Daniel Moun...|           Smith Inc|    1|        2014|
    |     Cynthia Norton|37.0|       9191.58|              0| 5.56|      9.0|2016-01-19 15:31:15|765 Tricia Row Ka...|          Love-Jones|    1|        2016|
    |   Jessica Williams|48.0|      10356.02|              0| 5.12|      8.0|2009-03-03 23:13:37|6187 Olson Mounta...|        Kelly-Warren|    1|        2009|
    |        Eric Butler|44.0|      11331.58|              1| 5.23|     11.0|2016-12-05 03:35:43|4846 Savannah Roa...|   Reynolds-Sheppard|    1|        2016|
    |      Zachary Walsh|32.0|       9885.12|              1| 6.92|      9.0|2006-03-09 14:50:20|25271 Roy Express...|          Singh-Cole|    1|        2006|
    |        Ashlee Carr|43.0|       14062.6|              1| 5.46|     11.0|2011-09-29 05:47:23|3725 Caroline Str...|           Lopez PLC|    1|        2011|
    |     Jennifer Lynch|40.0|       8066.94|              1| 7.11|     11.0|2006-03-28 15:42:45|363 Sandra Lodge ...|       Reed-Martinez|    1|        2006|
    |       Paula Harris|30.0|      11575.37|              1| 5.22|      8.0|2016-11-13 13:13:01|Unit 8120 Box 916...|Briggs, Lamb and ...|    1|        2016|
    |     Bruce Phillips|45.0|       8771.02|              1| 6.64|     11.0|2015-05-28 12:14:03|Unit 1895 Box 094...|    Figueroa-Maynard|    1|        2015|
    |       Craig Garner|45.0|       8988.67|              1| 4.84|     11.0|2011-02-16 08:10:47|897 Kelley Overpa...|     Abbott-Thompson|    1|        2011|
    |       Nicole Olson|40.0|       8283.32|              1|  5.1|     13.0|2012-11-22 05:35:03|11488 Weaver Cape...|Smith, Kim and Ma...|    1|        2012|
    |     Harold Griffin|41.0|       6569.87|              1|  4.3|     11.0|2015-03-28 02:13:44|1774 Peter Row Ap...|Snyder, Lee and M...|    1|        2015|
    |       James Wright|38.0|      10494.82|              1| 6.81|     12.0|2015-07-22 08:38:40|45408 David Path ...|      Sanders-Pierce|    1|        2015|
    |      Doris Wilkins|45.0|       8213.41|              1| 7.35|     11.0|2006-09-03 06:13:55|28216 Wright Moun...|Andrews, Adams an...|    1|        2006|
    |Katherine Carpenter|43.0|      11226.88|              0| 8.08|     12.0|2006-10-22 04:42:38|Unit 4948 Box 481...|Morgan, Phillips ...|    1|        2006|
    |     Lindsay Martin|53.0|       5515.09|              0| 6.85|      8.0|2015-10-07 00:27:10|69203 Crosby Divi...|      Villanueva LLC|    1|        2015|
    |        Kathy Curry|46.0|        8046.4|              1| 5.69|      8.0|2014-11-06 23:47:14|9569 Caldwell Cre...|Berry, Orr and Ca...|    1|        2014|
    +-------------------+----+--------------+---------------+-----+---------+-------------------+--------------------+--------------------+-----+------------+
    only showing top 20 rows
    



```python
my_cols = df.select('Age','Total_Purchase','Years','Num_Sites','Onboard_Year','Location','Company','Churn')
my_cols.show()
```

    +----+--------------+-----+---------+------------+--------------------+--------------------+-----+
    | Age|Total_Purchase|Years|Num_Sites|Onboard_Year|            Location|             Company|Churn|
    +----+--------------+-----+---------+------------+--------------------+--------------------+-----+
    |42.0|       11066.8| 7.22|      8.0|        2013|10265 Elizabeth M...|          Harvey LLC|    1|
    |41.0|      11916.22|  6.5|     11.0|        2013|6157 Frank Garden...|          Wilson PLC|    1|
    |38.0|      12884.75| 6.67|     12.0|        2016|1331 Keith Court ...|Miller, Johnson a...|    1|
    |42.0|       8010.76| 6.71|     10.0|        2014|13120 Daniel Moun...|           Smith Inc|    1|
    |37.0|       9191.58| 5.56|      9.0|        2016|765 Tricia Row Ka...|          Love-Jones|    1|
    |48.0|      10356.02| 5.12|      8.0|        2009|6187 Olson Mounta...|        Kelly-Warren|    1|
    |44.0|      11331.58| 5.23|     11.0|        2016|4846 Savannah Roa...|   Reynolds-Sheppard|    1|
    |32.0|       9885.12| 6.92|      9.0|        2006|25271 Roy Express...|          Singh-Cole|    1|
    |43.0|       14062.6| 5.46|     11.0|        2011|3725 Caroline Str...|           Lopez PLC|    1|
    |40.0|       8066.94| 7.11|     11.0|        2006|363 Sandra Lodge ...|       Reed-Martinez|    1|
    |30.0|      11575.37| 5.22|      8.0|        2016|Unit 8120 Box 916...|Briggs, Lamb and ...|    1|
    |45.0|       8771.02| 6.64|     11.0|        2015|Unit 1895 Box 094...|    Figueroa-Maynard|    1|
    |45.0|       8988.67| 4.84|     11.0|        2011|897 Kelley Overpa...|     Abbott-Thompson|    1|
    |40.0|       8283.32|  5.1|     13.0|        2012|11488 Weaver Cape...|Smith, Kim and Ma...|    1|
    |41.0|       6569.87|  4.3|     11.0|        2015|1774 Peter Row Ap...|Snyder, Lee and M...|    1|
    |38.0|      10494.82| 6.81|     12.0|        2015|45408 David Path ...|      Sanders-Pierce|    1|
    |45.0|       8213.41| 7.35|     11.0|        2006|28216 Wright Moun...|Andrews, Adams an...|    1|
    |43.0|      11226.88| 8.08|     12.0|        2006|Unit 4948 Box 481...|Morgan, Phillips ...|    1|
    |53.0|       5515.09| 6.85|      8.0|        2015|69203 Crosby Divi...|      Villanueva LLC|    1|
    |46.0|        8046.4| 5.69|      8.0|        2014|9569 Caldwell Cre...|Berry, Orr and Ca...|    1|
    +----+--------------+-----+---------+------------+--------------------+--------------------+-----+
    only showing top 20 rows
    



```python
new_data = my_cols.na.drop()
new_data.summary().show()
```

    +-------+-----------------+-----------------+-----------------+------------------+------------------+--------------------+--------------------+-------------------+
    |summary|              Age|   Total_Purchase|            Years|         Num_Sites|      Onboard_Year|            Location|             Company|              Churn|
    +-------+-----------------+-----------------+-----------------+------------------+------------------+--------------------+--------------------+-------------------+
    |  count|              900|              900|              900|               900|               900|                 900|                 900|                900|
    |   mean|41.81666666666667|10062.82403333334| 5.27315555555555| 8.587777777777777|2010.8011111111111|                null|                null|0.16666666666666666|
    | stddev|6.127560416916251|2408.644531858096|1.274449013194616|1.7648355920350969|3.2072288498508783|                null|                null| 0.3728852122772358|
    |    min|             22.0|            100.0|              1.0|               3.0|              2006|00103 Jeffrey Cre...|     Abbott-Thompson|                  0|
    |    25%|             38.0|          8480.93|             4.45|               7.0|              2008|                null|                null|                  0|
    |    50%|             42.0|         10041.13|             5.21|               8.0|              2011|                null|                null|                  0|
    |    75%|             46.0|         11758.69|             6.11|              10.0|              2014|                null|                null|                  0|
    |    max|             65.0|         18026.01|             9.15|              14.0|              2016|Unit 9800 Box 287...|Zuniga, Clark and...|                  1|
    +-------+-----------------+-----------------+-----------------+------------------+------------------+--------------------+--------------------+-------------------+
    



```python
from pyspark.sql.functions import countDistinct, count_distinct
```


```python
df.select(countDistinct('Location')).show()
```

    +------------------------+
    |count(DISTINCT Location)|
    +------------------------+
    |                     900|
    +------------------------+
    



```python
df.select(countDistinct('Company')).show()
```

    +-----------------------+
    |count(DISTINCT Company)|
    +-----------------------+
    |                    873|
    +-----------------------+
    



```python
df.select('Age','Total_Purchase','Years','Num_Sites','Onboard_Year','Churn').show()
```

    +----+--------------+-----+---------+------------+-----+
    | Age|Total_Purchase|Years|Num_Sites|Onboard_Year|Churn|
    +----+--------------+-----+---------+------------+-----+
    |42.0|       11066.8| 7.22|      8.0|        2013|    1|
    |41.0|      11916.22|  6.5|     11.0|        2013|    1|
    |38.0|      12884.75| 6.67|     12.0|        2016|    1|
    |42.0|       8010.76| 6.71|     10.0|        2014|    1|
    |37.0|       9191.58| 5.56|      9.0|        2016|    1|
    |48.0|      10356.02| 5.12|      8.0|        2009|    1|
    |44.0|      11331.58| 5.23|     11.0|        2016|    1|
    |32.0|       9885.12| 6.92|      9.0|        2006|    1|
    |43.0|       14062.6| 5.46|     11.0|        2011|    1|
    |40.0|       8066.94| 7.11|     11.0|        2006|    1|
    |30.0|      11575.37| 5.22|      8.0|        2016|    1|
    |45.0|       8771.02| 6.64|     11.0|        2015|    1|
    |45.0|       8988.67| 4.84|     11.0|        2011|    1|
    |40.0|       8283.32|  5.1|     13.0|        2012|    1|
    |41.0|       6569.87|  4.3|     11.0|        2015|    1|
    |38.0|      10494.82| 6.81|     12.0|        2015|    1|
    |45.0|       8213.41| 7.35|     11.0|        2006|    1|
    |43.0|      11226.88| 8.08|     12.0|        2006|    1|
    |53.0|       5515.09| 6.85|      8.0|        2015|    1|
    |46.0|        8046.4| 5.69|      8.0|        2014|    1|
    +----+--------------+-----+---------+------------+-----+
    only showing top 20 rows
    



```python
from pyspark.ml.feature import VectorAssembler
```


```python
assembler = VectorAssembler(inputCols=['Age','Total_Purchase','Years','Num_Sites','Onboard_Year'],
                           outputCol='features')
```


```python
output = assembler.transform(df)
```


```python
final_data = output.select('features','churn')
final_data.show()
```

    +--------------------+-----+
    |            features|churn|
    +--------------------+-----+
    |[42.0,11066.8,7.2...|    1|
    |[41.0,11916.22,6....|    1|
    |[38.0,12884.75,6....|    1|
    |[42.0,8010.76,6.7...|    1|
    |[37.0,9191.58,5.5...|    1|
    |[48.0,10356.02,5....|    1|
    |[44.0,11331.58,5....|    1|
    |[32.0,9885.12,6.9...|    1|
    |[43.0,14062.6,5.4...|    1|
    |[40.0,8066.94,7.1...|    1|
    |[30.0,11575.37,5....|    1|
    |[45.0,8771.02,6.6...|    1|
    |[45.0,8988.67,4.8...|    1|
    |[40.0,8283.32,5.1...|    1|
    |[41.0,6569.87,4.3...|    1|
    |[38.0,10494.82,6....|    1|
    |[45.0,8213.41,7.3...|    1|
    |[43.0,11226.88,8....|    1|
    |[53.0,5515.09,6.8...|    1|
    |[46.0,8046.4,5.69...|    1|
    +--------------------+-----+
    only showing top 20 rows
    



```python
final_data.select(countDistinct('churn')).show()
```

    +---------------------+
    |count(DISTINCT churn)|
    +---------------------+
    |                    2|
    +---------------------+
    



```python
train_churn,test_churn = final_data.randomSplit([0.7,0.3])
```


```python
train_churn.describe().show()
```

    +-------+-------------------+
    |summary|              churn|
    +-------+-------------------+
    |  count|                638|
    |   mean|0.17398119122257052|
    | stddev|0.37939075297736585|
    |    min|                  0|
    |    max|                  1|
    +-------+-------------------+
    



```python
test_churn.describe().show()
```

    +-------+-------------------+
    |summary|              churn|
    +-------+-------------------+
    |  count|                262|
    |   mean|0.14885496183206107|
    | stddev|0.35662668423738547|
    |    min|                  0|
    |    max|                  1|
    +-------+-------------------+
    



```python
from pyspark.ml.classification import LogisticRegression
```


```python
log_reg_data = LogisticRegression(featuresCol='features', labelCol='churn')
```


```python
fit_model = log_reg_data.fit(train_churn)
```


```python
training_summary = fit_model.summary
training_summary.predictions.describe().show()
```

    +-------+-------------------+-------------------+
    |summary|              churn|         prediction|
    +-------+-------------------+-------------------+
    |  count|                638|                638|
    |   mean|0.17398119122257052|0.13166144200626959|
    | stddev|0.37939075297736585|0.33838762310655013|
    |    min|                0.0|                0.0|
    |    max|                1.0|                1.0|
    +-------+-------------------+-------------------+
    



```python
results = fit_model.transform(test_churn)
```


```python
results.show()
```

    +--------------------+-----+--------------------+--------------------+----------+
    |            features|churn|       rawPrediction|         probability|prediction|
    +--------------------+-----+--------------------+--------------------+----------+
    |[22.0,11254.38,4....|    0|[4.84512177001655...|[0.99219474183468...|       0.0|
    |[26.0,8787.39,5.4...|    1|[0.80088724985082...|[0.69016424047663...|       0.0|
    |[28.0,8670.98,3.9...|    0|[7.62650878077064...|[0.99951287803337...|       0.0|
    |[28.0,11204.23,3....|    0|[1.82706428211059...|[0.86141162773245...|       0.0|
    |[28.0,11245.38,6....|    0|[3.23727761943303...|[0.96221325115664...|       0.0|
    |[29.0,8688.17,5.7...|    1|[2.74609552777563...|[0.93969246077251...|       0.0|
    |[29.0,11274.46,4....|    0|[4.72051484054086...|[0.99116810753677...|       0.0|
    |[30.0,7960.64,2.7...|    1|[3.46175502534318...|[0.96957977360242...|       0.0|
    |[30.0,8874.83,5.5...|    0|[2.97435221511394...|[0.95140190321586...|       0.0|
    |[30.0,10183.98,5....|    0|[3.15931400408344...|[0.95927415489611...|       0.0|
    |[30.0,10960.52,5....|    0|[2.65301667613227...|[0.93419667845695...|       0.0|
    |[30.0,13473.35,3....|    0|[2.76418768736397...|[0.94070963388157...|       0.0|
    |[31.0,7073.61,5.7...|    0|[2.74471058021462...|[0.93961392723592...|       0.0|
    |[31.0,9574.89,7.3...|    0|[2.83405396177168...|[0.94448853445454...|       0.0|
    |[32.0,8617.98,6.2...|    1|[1.22704509812901...|[0.77330097891823...|       0.0|
    |[32.0,10716.75,5....|    0|[4.27401303649074...|[0.98626547783750...|       0.0|
    |[32.0,12479.72,4....|    0|[4.33668972732550...|[0.98708911700381...|       0.0|
    |[32.0,12547.91,7....|    0|[0.05539247390267...|[0.51384457868309...|       0.0|
    |[32.0,13630.93,4....|    0|[2.19645550271955...|[0.89993076198823...|       0.0|
    |[33.0,7492.9,6.71...|    0|[4.32777827658944...|[0.98697505340611...|       0.0|
    +--------------------+-----+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
results.select('churn','prediction').show()
```

    +-----+----------+
    |churn|prediction|
    +-----+----------+
    |    0|       0.0|
    |    1|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    1|       0.0|
    |    0|       0.0|
    |    1|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    1|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    |    0|       0.0|
    +-----+----------+
    only showing top 20 rows
    



```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
prediction_and_labels = fit_model.evaluate(test_churn)
```


```python
churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='churn')
```


```python
auc = churn_eval.evaluate(prediction_and_labels.predictions)
auc
```




    0.6871909853972633




```python
#let's see how it fits on new data and what results we will take
```


```python
final_log_reg_model = log_reg_data.fit(final_data)
```


```python
new_customers = spark.read.csv('new_customers.csv',inferSchema=True,header=True)
```


```python
#check if the new_customers is in the same format
new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
    



```python
new_customers.describe().show()
```

    +-------+-------------+------------------+-----------------+------------------+-----------------+------------------+--------------------+----------------+
    |summary|        Names|               Age|   Total_Purchase|   Account_Manager|            Years|         Num_Sites|            Location|         Company|
    +-------+-------------+------------------+-----------------+------------------+-----------------+------------------+--------------------+----------------+
    |  count|            6|                 6|                6|                 6|                6|                 6|                   6|               6|
    |   mean|         null|35.166666666666664|7607.156666666667|0.8333333333333334|6.808333333333334|12.333333333333334|                null|            null|
    | stddev|         null| 15.71517313511584|4346.008232825459| 0.408248290463863|3.708737880555414|3.3862466931200785|                null|            null|
    |    min|Andrew Mccall|              22.0|            100.0|                 0|              1.0|               8.0|085 Austin Views ...|Barron-Robertson|
    |    max| Taylor Young|              65.0|         13147.71|                 1|             10.0|              15.0|Unit 0789 Box 073...|        Wood LLC|
    +-------+-------------+------------------+-----------------+------------------+-----------------+------------------+--------------------+----------------+
    



```python
new_customers = new_customers.na.drop()
new_customers.show()
```

    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+
    |         Names| Age|Total_Purchase|Account_Manager|Years|Num_Sites|       Onboard_date|            Location|         Company|
    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+
    | Andrew Mccall|37.0|       9935.53|              1| 7.71|      8.0|2011-08-29 18:37:54|38612 Johnny Stra...|        King Ltd|
    |Michele Wright|23.0|       7526.94|              1| 9.28|     15.0|2013-07-22 18:19:54|21083 Nicole Junc...|   Cannon-Benson|
    |  Jeremy Chang|65.0|         100.0|              1|  1.0|     15.0|2006-12-11 07:48:13|085 Austin Views ...|Barron-Robertson|
    |Megan Ferguson|32.0|        6487.5|              0|  9.4|     14.0|2016-10-28 05:32:13|922 Wright Branch...|   Sexton-Golden|
    |  Taylor Young|32.0|      13147.71|              1| 10.0|      8.0|2012-03-20 00:36:46|Unit 0789 Box 073...|        Wood LLC|
    | Jessica Drake|22.0|       8445.26|              1| 3.46|     14.0|2011-02-04 19:29:27|1148 Tina Straven...|   Parks-Robbins|
    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+
    



```python
new_customers = new_customers.withColumn('Onboard_Year', year(new_customers['Onboard_date']))
```


```python
new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Onboard_Year: integer (nullable = true)
    



```python
test_new_customers = assembler.transform(new_customers)
```


```python
test_new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Onboard_Year: integer (nullable = true)
     |-- features: vector (nullable = true)
    



```python
final_results = final_log_reg_model.transform(test_new_customers)
```


```python
final_results.show()
```

    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+------------+--------------------+--------------------+--------------------+----------+
    |         Names| Age|Total_Purchase|Account_Manager|Years|Num_Sites|       Onboard_date|            Location|         Company|Onboard_Year|            features|       rawPrediction|         probability|prediction|
    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+------------+--------------------+--------------------+--------------------+----------+
    | Andrew Mccall|37.0|       9935.53|              1| 7.71|      8.0|2011-08-29 18:37:54|38612 Johnny Stra...|        King Ltd|        2011|[37.0,9935.53,7.7...|[2.42135286076788...|[0.91844114068003...|       0.0|
    |Michele Wright|23.0|       7526.94|              1| 9.28|     15.0|2013-07-22 18:19:54|21083 Nicole Junc...|   Cannon-Benson|        2013|[23.0,7526.94,9.2...|[-5.9772226044142...|[0.00252944528770...|       1.0|
    |  Jeremy Chang|65.0|         100.0|              1|  1.0|     15.0|2006-12-11 07:48:13|085 Austin Views ...|Barron-Robertson|        2006|[65.0,100.0,1.0,1...|[-3.4636260880724...|[0.03036508817007...|       1.0|
    |Megan Ferguson|32.0|        6487.5|              0|  9.4|     14.0|2016-10-28 05:32:13|922 Wright Branch...|   Sexton-Golden|        2016|[32.0,6487.5,9.4,...|[-5.3152623064811...|[0.00489194036494...|       1.0|
    |  Taylor Young|32.0|      13147.71|              1| 10.0|      8.0|2012-03-20 00:36:46|Unit 0789 Box 073...|        Wood LLC|        2012|[32.0,13147.71,10...|[1.29647320283950...|[0.78524083051707...|       0.0|
    | Jessica Drake|22.0|       8445.26|              1| 3.46|     14.0|2011-02-04 19:29:27|1148 Tina Straven...|   Parks-Robbins|        2011|[22.0,8445.26,3.4...|[-1.4416030196523...|[0.19129723452150...|       1.0|
    +--------------+----+--------------+---------------+-----+---------+-------------------+--------------------+----------------+------------+--------------------+--------------------+--------------------+----------+
    



```python
final_results.select('Company','Names','prediction').show()
```

    +----------------+--------------+----------+
    |         Company|         Names|prediction|
    +----------------+--------------+----------+
    |        King Ltd| Andrew Mccall|       0.0|
    |   Cannon-Benson|Michele Wright|       1.0|
    |Barron-Robertson|  Jeremy Chang|       1.0|
    |   Sexton-Golden|Megan Ferguson|       1.0|
    |        Wood LLC|  Taylor Young|       0.0|
    |   Parks-Robbins| Jessica Drake|       1.0|
    +----------------+--------------+----------+
    



```python

```
