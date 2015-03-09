package org.apache.mahout.sparkbindings.drm

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{Text, IntWritable, SequenceFile}

import org.apache.mahout.sparkbindings.SparkEngine._

import org.apache.mahout.math.drm.DrmLikeOpsSuiteBase
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.FunSuite

/**
 * @author gokhan
 */
class DrmLikeIOSuite extends FunSuite with DistributedSparkSuite with DrmLikeOpsSuiteBase {
  test("seq2sparse") {
    val textCollection = createTextCollection(TmpDir)
    val fileName = textCollection.toUri.getPath


    val llr = -1.0

    val matrixAndDictTFUni = drmDfsTextRead(path = fileName, weight = "tf")
    val matrixAndDictTFIDFUni = drmDfsTextRead(path = fileName, weight = "tfidf")
    val matrixAndDictTFBi = drmDfsTextRead(path = fileName, weight = "tf",
      allowedNgrams = 2 to 2, minLLR = llr)
    val matrixAndDictTFIDFBi = drmDfsTextRead(path = fileName, weight = "tfidf",
      allowedNgrams = 2 to 2, minLLR = llr)

    val dictUni = matrixAndDictTFUni._2.toMap
    val dictBi = matrixAndDictTFBi._2.toMap


    dictUni should equal(matrixAndDictTFIDFUni._2.toMap)
    dictBi should equal(matrixAndDictTFIDFBi._2.toMap)

    val actualTfs1:Map[Int, Map[String, Double]] = Map(0->Map("word1"->1.0, "word2"->2, "word3"->1), 1->Map("word1"->1, "word4"->2, "word3"->1),
      3->Map("word1"->1, "word2"->1, "word3"->1, "word4"->1))

    val actualTfs2:Map[Int, Map[String, Double]] = Map(0->Map("word1 word2"->1.0, "word2 word3"->1, "word3 word2"->1),
      1->Map("word1 word4"->1,
        "word4 word3"->2,
        "word3 word4"->1),
      3->Map("word1 word2"->1.0, "word2 word3"->1, "word3 word4"->1))

    val n = 4
    val actualIdfs1 = Map("word1"->idf(n, 3), "word2"->idf(n, 2), "word3"->idf(n, 3), "word4"->idf(n, 2))
    val actualIdfs2 = Map("word1 word2"->idf(n, 2), "word2 word3"->idf(n, 2), "word3 word2"->idf(n, 1), "word1 word4"->idf(n, 1),
      "word4 word3"->idf(n, 1), "word3 word4"->idf(n, 2))

    val tfUni = matrixAndDictTFUni._1.collect
    val tfIdfUni = matrixAndDictTFIDFUni._1.collect
    val tfBi = matrixAndDictTFBi._1.collect
    val tfIdfBi = matrixAndDictTFIDFBi._1.collect

    tfUni.get(0, dictUni("word1")) should equal(actualTfs1(0)("word1"))
    tfUni.get(0, dictUni("word2")) should equal(actualTfs1(0)("word2"))
    tfUni.get(0, dictUni("word3")) should equal(actualTfs1(0)("word3"))
    tfUni.get(3, dictUni("word1")) should equal(actualTfs1(3)("word1"))
    tfUni.get(3, dictUni("word2")) should equal(actualTfs1(3)("word2"))
    tfUni.get(3, dictUni("word4")) should equal(actualTfs1(3)("word4"))

    tfIdfUni.get(0, dictUni("word1")) should equal(actualTfs1(0)("word1")*actualIdfs1("word1"))
    tfIdfUni.get(0, dictUni("word2")) should equal(actualTfs1(0)("word2")*actualIdfs1("word2"))
    tfIdfUni.get(0, dictUni("word3")) should equal(actualTfs1(0)("word3")*actualIdfs1("word3"))
    tfIdfUni.get(3, dictUni("word1")) should equal(actualTfs1(3)("word1")*actualIdfs1("word1"))
    tfIdfUni.get(3, dictUni("word2")) should equal(actualTfs1(3)("word2")*actualIdfs1("word2"))
    tfIdfUni.get(3, dictUni("word4")) should equal(actualTfs1(3)("word4")*actualIdfs1("word4"))

    tfBi.get(0, dictBi("word1 word2")) should equal(actualTfs2(0)("word1 word2"))
    tfBi.get(0, dictBi("word2 word3")) should equal(actualTfs2(0)("word2 word3"))
    tfBi.get(0, dictBi("word3 word2")) should equal(actualTfs2(0)("word3 word2"))
    tfBi.get(3, dictBi("word1 word2")) should equal(actualTfs2(3)("word1 word2"))
    tfBi.get(3, dictBi("word2 word3")) should equal(actualTfs2(3)("word2 word3"))
    tfBi.get(3, dictBi("word3 word4")) should equal(actualTfs2(3)("word3 word4"))

    tfIdfBi.get(0, dictBi("word1 word2")) should equal(actualTfs2(0)("word1 word2")*actualIdfs2("word1 word2"))
    tfIdfBi.get(0, dictBi("word2 word3")) should equal(actualTfs2(0)("word2 word3")*actualIdfs2("word2 word3"))
    tfIdfBi.get(0, dictBi("word3 word2")) should equal(actualTfs2(0)("word3 word2")*actualIdfs2("word3 word2"))
    tfIdfBi.get(3, dictBi("word1 word2")) should equal(actualTfs2(3)("word1 word2")*actualIdfs2("word1 word2"))
    tfIdfBi.get(3, dictBi("word2 word3")) should equal(actualTfs2(3)("word2 word3")*actualIdfs2("word2 word3"))
    tfIdfBi.get(3, dictBi("word3 word4")) should equal(actualTfs2(3)("word3 word4")*actualIdfs2("word3 word4"))

  }

  private def createTextCollection(tmpDir: String): Path = {
    val conf = new Configuration()

    val path = new Path(tmpDir+"text")
    val seqWriter = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
      SequenceFile.Writer.keyClass(classOf[IntWritable]),
      SequenceFile.Writer.valueClass(classOf[Text]))

    seqWriter.append(new IntWritable(0), new Text("Word1 word2 word3 word2"))
    seqWriter.append(new IntWritable(1), new Text("Word1 word4 word3 word4"))
    seqWriter.append(new IntWritable(3), new Text("Word1 word2 word3 word4"))
    seqWriter.close()

    path
  }

  private def idf(n:Int, df:Double):Double = Math.log(n/df)

}
