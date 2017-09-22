package com.metistream.claims


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.types.DoubleType
//import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StructField}
import org.apache.spark.sql.functions.udf

/**
  * Created by eliu on 9/20/17.
  */
object diabetesModeling {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sparkContext = new SparkContext(conf)
    val sQLContext = new SQLContext(sparkContext)
//    val spark = SparkSession.builder().master("local").getOrCreate()

    val toDouble = udf[Double, String](_.toDouble)

    var summaryDF = sQLContext.read.format("jdbc")
      .option("url", "jdbc:mysql://35.188.191.104:3306/claims")
      .option("dbtable", "DE1_0_2009_Beneficiary_Summary_File_Sample_1")
      .option("user", "root")
      .option("password", "1234")
      .option("driver", "com.mysql.jdbc.Driver")
      .load()
    //create column birthYear based on BENE_BIRTH_DT and cast to integer
    summaryDF = summaryDF.withColumn("birthYear", summaryDF("BENE_BIRTH_DT").substr(1, 4))
    summaryDF = summaryDF.withColumn("birthYear", toDouble(summaryDF("birthYear")))

    //create column birthMonth based on BENE_BIRTH_DT and cast to integer
    summaryDF = summaryDF.withColumn("birthMonth", summaryDF("BENE_BIRTH_DT").substr(5, 2))
    summaryDF = summaryDF.withColumn("birthMonth", summaryDF("birthMonth").cast(IntegerType))
//      .withColumn("BENE_SEX_IDENT_CD", toDouble(summaryDF("BENE_SEX_IDENT_CD")))
//      .withColumn("BENE_RACE_CD", toDouble(summaryDF("BENE_RACE_CD")))
//      .withColumn("SP_STATE_CODE", toDouble(summaryDF("SP_STATE_CODE")))
//      .withColumn("BENE_COUNTY_CD", toDouble(summaryDF("BENE_COUNTY_CD")))
//      .withColumn("SP_ALZHDMTA", toDouble(summaryDF("SP_ALZHDMTA")))
//      .withColumn("SP_CHF", toDouble(summaryDF("SP_CHF")))
//      .withColumn("SP_CHRNKIDN", toDouble(summaryDF("SP_CHRNKIDN")))
//      .withColumn("SP_CNCR", toDouble(summaryDF("SP_CNCR")))
//      .withColumn("SP_COPD", toDouble(summaryDF("SP_COPD")))
//      .withColumn("SP_DEPRESSN", toDouble(summaryDF("SP_DEPRESSN")))
//      .withColumn("SP_ISCHMCHT", toDouble(summaryDF("SP_ISCHMCHT")))
//      .withColumn("SP_OSTEOPRS", toDouble(summaryDF("SP_OSTEOPRS")))
//      .withColumn("SP_RA_OA", toDouble(summaryDF("SP_RA_OA")))
//      .withColumn("SP_STRKETIA", toDouble(summaryDF("SP_STRKETIA")))




    summaryDF.select("birthYear", "birthMonth").show()

    summaryDF.printSchema()

    val columns = summaryDF.columns

    val categoricalColumns = Array(
      "BENE_SEX_IDENT_CD",
      "BENE_RACE_CD",
      "SP_STATE_CODE",
//      "BENE_COUNTY_CD",
      "SP_ALZHDMTA",
      "SP_CHF",
      "SP_CHRNKIDN",
      "SP_CNCR",
      "SP_COPD",
      "SP_DEPRESSN",
      "SP_ISCHMCHT",
      "SP_OSTEOPRS",
      "SP_RA_OA",
      "SP_STRKETIA",
      "birthMonth"
    )

//    val categoricalColumns = Array(
//      "BENE_RACE_CD")

    val catFeatureIndexers = categoricalColumns.map{x =>
      new StringIndexer().setInputCol(x).setOutputCol(x + "_index")
    }

    val catFeatureOneHotEncoders = catFeatureIndexers
      .map(x => new OneHotEncoder().setInputCol(x.getOutputCol).setOutputCol(s"${x.getOutputCol}_oh"))


    val featureNames = Array(
      "BENE_HI_CVRAGE_TOT_MONS",
      "BENE_SMI_CVRAGE_TOT_MONS",
      "BENE_HMO_CVRAGE_TOT_MONS",
      "PLAN_CVRG_MOS_NUM",
      "MEDREIMB_IP",
      "BENRES_IP",
      "PPPYMT_IP",
      "MEDREIMB_OP",
      "BENRES_OP",
      "PPPYMT_OP",
      "MEDREIMB_CAR",
      "BENRES_CAR",
      "PPPYMT_CAR",
      "birthYear"
    ).union(catFeatureOneHotEncoders.map(_.getOutputCol))

//    val featureNames = Array(
//      "BENE_HI_CVRAGE_TOT_MONS"
//    ).union(catFeatureOneHotEncoders.map(_.getOutputCol))

    println("features: " + featureNames.mkString(","))

    val featureAssembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")

    val labelIndexer = new StringIndexer().setInputCol("SP_DIABETES").setOutputCol("label")


    val lr = new LogisticRegression()
      .setMaxIter(2)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val Array(trainingData, testData) = summaryDF.randomSplit(Array(0.8, 0.2))

    val pipeline = new Pipeline().setStages(catFeatureIndexers.union(catFeatureOneHotEncoders).union(Array(featureAssembler, labelIndexer, lr))).fit(trainingData)

    val prediction = pipeline.transform(testData)


    prediction.select("probability", "prediction", "label", "SP_DIABETES").collect().foreach(println)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction")

    println("Accuracy=========== " + evaluator.evaluate(prediction))

    println(summaryDF.schema.json)

    saveToModelFile(summaryDF, pipeline, "/Users/eliu/Desktop/claims_diabetes_logistic_regression.txt")

  }


  def saveToModelFile(dataframe: org.apache.spark.sql.DataFrame, pipelineModel: org.apache.spark.ml.PipelineModel, filePath: String) = {
    import java.io._
    import java.util._

    import org.apache.spark.sql.functions._

    val model = pipelineModel.stages.filter(_.hasParam("labelCol")).head

    //     val labelCol = model.extractParamMap().get(new org.apache.spark.ml.param.Param(model.uid, "labelCol","d")).get
    val algorithm = model.getClass.toString().split(" ")(1).split("\\.").takeRight(2).mkString(",")

    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)
    oos.writeObject(pipelineModel)
    oos.close()
    val modelString = Base64.getEncoder().encodeToString(baos.toByteArray())

    val categoryFeatures = "{" + dataframe.schema.toArray.collect {
      case StructField(a, b, c, d) if (b.typeName == "string") =>
        val col = "\"" + a + "\":"
        val arr = "[" + dataframe.select(a).distinct.rdd.collect().map(x => "\"" + x.getAs[String](0) + "\"").mkString(",") + "]"
        col + arr
      case StructField(a, b, c, d) if b.typeName == "double" || b.typeName == "integer" =>
        val col = "\"" + a + "\":"
        val arr = "[" + dataframe.select(min(a), max(a)).rdd.collect()(0).mkString(",") + "]"
        col + arr

    }.mkString(",") + "}"


    val schema = dataframe.schema.json

    val writer = new PrintWriter(new File(filePath))
    val result = schema + "|" + modelString + "|" + categoryFeatures + "|" + model.extractParamMap().get(new org.apache.spark.ml.param.Param(model.uid, "labelCol", "d")).get + "|" + algorithm
    writer.write(result)
    writer.close()
  }
}


// feature indexed -> feature OHE -> feature assembled -> label indexed -> label assembled