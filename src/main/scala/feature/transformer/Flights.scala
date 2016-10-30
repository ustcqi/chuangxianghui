package feature.transformer

import common.AppContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._

/**
  * Data: http://stat-computing.org/dataexpo/2009/the-data.html
  * Created by kerwin on 16/10/7.
  */
class Flights(context: AppContext) extends Serializable{
    @transient
    val sc = context.sc
    @transient
    val sqlContext = context.sqlContext

    val trainingFile = "./data/flights/2007.csv"
    val testFile = "./data/flights/2007_test.csv"

    val schema = StructType(
        StructField("Month", StringType, true)
                :: StructField("DayOfMonth", StringType, true)
                :: StructField("DayOfWeek", StringType, true)
                :: StructField("DepTime", IntegerType, true)
                :: StructField("CRSDepTime", IntegerType, true)

                :: StructField("ArrTime", IntegerType, true)
                :: StructField("CRSArrTime", IntegerType, true)
                :: StructField("UniqueCarrier", StringType, true)
                :: StructField("ActualElapsedTime", IntegerType, true)
                :: StructField("CRSElapsedTime", IntegerType, true)

                :: StructField("AirTime", IntegerType, true)
                :: StructField("ArrDelay", DoubleType, true)
                :: StructField("DepDelay", IntegerType, true)
                :: StructField("Origin", StringType, true)
                :: StructField("Distance", IntegerType, true)
                :: Nil)

    def getMinuteOfDay(t : String) : Int = {
        val time = t.toInt
        time/100 * 60 + time%100
    }

    def getData(trainingFile: String, testFile: String): (DataFrame, DataFrame) ={
        val flight2007Rdd = sc.textFile(trainingFile)
        val header = flight2007Rdd.first
        val trainingData = flight2007Rdd.filter(x => x != header)
                .map(x => x.split(","))
                .filter(x => x(21) == "0")
                .filter(x => x(17) == "ORD")
                .filter(x => x(14) != "NA")
                .map(p => Row(p(1), p(2), p(3), getMinuteOfDay(p(4)), getMinuteOfDay(p(5)),
                        getMinuteOfDay(p(6)), getMinuteOfDay(p(7)), p(8), p(11).toInt, p(12).toInt,
                        p(13).toInt, p(14).toDouble, p(15).toInt, p(16), p(18).toInt)
                )
        val trainingDataDF = sqlContext.createDataFrame(trainingData, schema)

        val flight2008Rdd = sc.textFile(testFile)
        val testheader = flight2008Rdd.first
        val testData = flight2008Rdd.filter(x => x != testheader)
                .map(x => x.split(","))
                .filter(x => x(21) == "0")
                .filter(x => x(17) == "ORD")
                .filter(x => x(14) != "NA")
                .map(p => Row(p(1), p(2), p(3), getMinuteOfDay(p(4)), getMinuteOfDay(p(5)),
                        getMinuteOfDay(p(6)), getMinuteOfDay(p(7)), p(8), p(11).toInt, p(12).toInt,
                        p(13).toInt, p(14).toDouble, p(15).toInt, p(16), p(18).toInt)
                )
        val testDataDF = sqlContext.createDataFrame(testData, schema)
        (trainingDataDF, testDataDF)
    }
    
    def train(trainingDataDF: DataFrame): CrossValidatorModel = {
        val monthIndexer = new StringIndexer().setInputCol("Month")
                .setOutputCol("MonthCat")
        val dayOfMonthIndexer = new StringIndexer().setInputCol("DayOfMonth")
                .setOutputCol("DayOfMonthCat")
        val originIndexer = new StringIndexer().setInputCol("Origin")
                .setOutputCol("OriginCat")
        val uniqueCarrierIndexer = new StringIndexer().setInputCol("UniqueCarrier")
                .setOutputCol("UniqueCarrierCat")
        val uniqueCarrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierCat")
                .setOutputCol("UniqueCarrierVec")
        val dayOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek")
                .setOutputCol("DayOfWeekCat")
        val dowOneHotEncoder = new OneHotEncoder()
                .setInputCol("DayOfWeekCat")
                .setOutputCol("DayOfWeekVec")
        val stdAssembler = new VectorAssembler()
                .setInputCols(Array("MonthCat", "DayOfMonthCat", "DayOfWeekCat",
                    "UniqueCarrierCat", /*"OriginCat", "DepTime",*/ "CRSDepTime", "ArrTime",
                    "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime", "AirTime",
                    "DepDelay", "Distance"))
                .setOutputCol("stdAssemFeatures")
        val stdSlice = new VectorSlicer().setInputCol("stdAssemFeatures")
                .setOutputCol("stdSliceFeatures")
                .setNames(Array("MonthCat", "DayOfMonthCat", "DayOfWeekCat", "UniqueCarrierCat",
                    /*"OriginCat", "DepTime",*/"CRSDepTime", "ArrTime", "CRSArrTime",
                    "ActualElapsedTime", "CRSElapsedTime", "AirTime", "DepDelay", "Distance"))
        val stdScaler = new StandardScaler().setInputCol("stdSliceFeatures")
                .setOutputCol("stdFeatures").setWithStd(true).setWithMean(true)

        val minMaxAssembler = new VectorAssembler()
                .setInputCols(Array("OriginCat", "DepTime"))
                .setOutputCol("minMaxAssemFeatures")
        val minMaxSlice = new VectorSlicer()
                .setInputCol("minMaxAssemFeatures")
                .setOutputCol("minMaxSliceFeatures")
                .setNames(Array("OriginCat", "DepTime"))
        val minMaxScaler = new MinMaxScaler()
                .setInputCol("minMaxSliceFeatures")
                .setOutputCol("minMaxFeatures")
                .setMax(1.0)
                .setMin(0.0)

        val assembler = new VectorAssembler()
                .setInputCols(Array("stdFeatures", "minMaxFeatures", "DayOfWeekVec", "UniqueCarrierVec"))
                .setOutputCol("features")

        val binarizer = new Binarizer()
                .setInputCol("ArrDelay")
                .setOutputCol("label")
                .setThreshold(15.0)

        val lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
        val lrPipeline = new Pipeline()
                .setStages(Array(monthIndexer, dayOfMonthIndexer,  originIndexer,
                    uniqueCarrierIndexer, uniqueCarrierEncoder,
                    dayOfWeekIndexer, dowOneHotEncoder,
                    stdAssembler, stdSlice, stdScaler,
                    minMaxAssembler, minMaxSlice, minMaxScaler,
                    assembler, binarizer, lr))

        val paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam, Array(0.02, 0.05, 0.1))
                .addGrid(lr.elasticNetParam, Array(0.02, 0.1, 0.3))
                .addGrid(lr.maxIter, Array(10, 20))
                .build()

        val cv = new CrossValidator()
                .setEstimator(lrPipeline)
                .setEvaluator(new BinaryClassificationEvaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10)
        val cvModel = cv.fit(trainingDataDF)
        cvModel
    }

    /**
      *
      * @param testDataDF
      * @param cvModel
      * @return
      */
    def evaluate(cvModel: CrossValidatorModel, testDataDF: DataFrame): Double = {
        val predictedDF = cvModel.transform(testDataDF)
        val evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("precision")
        val accuracy = evaluator.evaluate(predictedDF)
        accuracy
    }

    def bestEstimatorParamMap(cvModel: CrossValidatorModel) : ParamMap = {
        cvModel.getEstimatorParamMaps
                .zip(cvModel.avgMetrics)
                .maxBy(_._2)
                ._1
    }
}

object Flights{
    def main(args: Array[String]): Unit = {
        val context = new AppContext()
        val flights = new Flights(context)
        val trainingFile = flights.trainingFile
        val testFile = flights.testFile
        val (trainingDataDF, testDataDF) = flights.getData(trainingFile, testFile)
        val cvModel = flights.train(trainingDataDF)
        val accuracy = flights.evaluate(cvModel, testDataDF)
        println("Error on test data = " + (1.0 - accuracy))
        //println(cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2)._1)
        println("Best Estimator Parameters:\n" + flights.bestEstimatorParamMap(cvModel))
        context.sc.stop()
    }
}

