package common

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by kerwin on 16/10/2.
  */
class AppContext {
    val appName = "sparkml"
    val conf = new SparkConf().setAppName(appName).setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
}
