/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.sparkbindings

import java.io.StringReader

import com.google.common.collect.{BiMap, HashBiMap}
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.mahout.common.lucene.AnalyzerUtils
import org.apache.mahout.drivers.TextDelimitedIndexedDatasetReader
import org.apache.mahout.math._
import org.apache.mahout.math.function.DoubleFunction
import org.apache.mahout.math.indexeddataset.{DefaultIndexedDatasetReadSchema, Schema, DefaultIndexedDatasetElementReadSchema}
import org.apache.mahout.math.stats.LogLikelihood
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.sparkbindings.drm.{CheckpointedDrmSpark, DrmRddInput}
import org.apache.mahout.math._
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import scala.reflect.ClassTag
import org.apache.spark.storage.StorageLevel
import org.apache.mahout.sparkbindings.blas._
import org.apache.hadoop.io._
import scala.Some
import scala.collection.JavaConversions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.spark.rdd.RDD
import org.apache.mahout.common.{Hadoop1HDFSUtil, HDFSUtil}

/** Spark-specific non-drm-method operations */
object SparkEngine extends DistributedEngine {

  // By default, use Hadoop 1 utils
  var hdfsUtils: HDFSUtil = Hadoop1HDFSUtil

  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd
      // Throw away keys
      .map(_._2)
      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter => {
      val acc = ((new DenseVector(n): Vector) /: iter)((acc, v) => acc += v)
      Iterator(acc)
    })
      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  def numNonZeroElementsPerColumn[K:ClassTag](drm: CheckpointedDrm[K]): Vector = {
    val n = drm.ncol

    drm.rdd
      // Throw away keys
      .map(_._2)
      // Fold() doesn't work with kryo still. So work around it.
      .mapPartitions(iter => {
      val acc = ((new DenseVector(n): Vector) /: iter) { (acc, v) =>
        v.nonZeroes().foreach { elem => acc(elem.index) += 1 }
        acc
      }
      Iterator(acc)
    })
      // Since we preallocated new accumulator vector per partition, this must not cause any side
      // effects now.
      .reduce(_ += _)
  }

  /** Engine-specific colMeans implementation based on a checkpoint. */
  override def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    if (drm.nrow == 0) drm.colSums() else drm.colSums() /= drm.nrow

  override def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double =
    drm.rdd
        // Compute sum of squares of each vector
        .map {
      case (key, v) => v dot v
    }
      .reduce(_ + _)


  /**
   * Perform default expression rewrite. Return physical plan that we can pass to exec(). <P>
   *
   * A particular physical engine implementation may choose to either use or not use these rewrites
   * as a useful basic rewriting rule.<P>
   */
  override def optimizerRewrite[K: ClassTag](action: DrmLike[K]): DrmLike[K] = super.optimizerRewrite(action)


  /** Second optimizer pass. Translate previously rewritten logical pipeline into physical engine plan. */
  def toPhysical[K: ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = {

    // Spark-specific Physical Plan translation.
    val rdd = tr2phys(plan)

    val newcp = new CheckpointedDrmSpark(
      rdd = rdd,
      _nrow = plan.nrow,
      _ncol = plan.ncol,
      _cacheStorageLevel = cacheHint2Spark(ch),
      partitioningTag = plan.partitioningTag
    )
    newcp.cache()
  }

  /** Broadcast support */
  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = dc.broadcast(v)

  /** Broadcast support */
  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = dc.broadcast(m)

  /**
   * Load DRM from hdfs (as in Mahout DRM format)
   *
   * @param path
   * @param sc spark context (wanted to make that implicit, doesn't work in current version of
   *           scala with the type bounds, sorry)
   *
   * @return DRM[Any] where Any is automatically translated to value type
   */
  def drmDfsRead (path: String, parMin:Int = 0)(implicit sc: DistributedContext): CheckpointedDrm[_] = {

    val drmMetadata = hdfsUtils.readDrmHeader(path)
    val k2vFunc = drmMetadata.keyW2ValFunc

    // Load RDD and convert all Writables to value types right away (due to reuse of writables in
    // Hadoop we must do it right after read operation).
    val rdd = sc.sequenceFile(path, classOf[Writable], classOf[VectorWritable], minPartitions = parMin)

        // Immediately convert keys and value writables into value types.
        .map { case (wKey, wVec) => k2vFunc(wKey) -> wVec.get()}

    // Wrap into a DRM type with correct matrix row key class tag evident.
    drmWrap(rdd = rdd, cacheHint = CacheHint.NONE)(drmMetadata.keyClassTag.asInstanceOf[ClassTag[Any]])
  }

  //currently, this only supports int keys. I couldn't manage to construct the DRM based on 'Any' keys that are
  // automatically translated, So I explicitly stated that the first RDD read from the seqfile is Int-keyed
  def drmDfsTextRead(path: String, weight:String = "tfidf", allowedNgrams:Range = 1 to 1, minDF:Int= 1,
                     maxDFSigma:Double = 3.0, maxDFPercent:Int = 100, minLLR:Double = 1.0,
                     parMin:Int = 0, analyzerClass:Class[_<:Analyzer] = classOf[StandardAnalyzer])
                    (implicit sc: DistributedContext): (CheckpointedDrm[_],Iterable[(String, Int)])= {

    import org.apache.spark.SparkContext._
    def tokenize[K:ClassTag](textRdd:RDD[(K, String)]):RDD[(K, Array[String])] = {

      textRdd.map{
        case (key, doc) =>
          val analyzer = AnalyzerUtils.createAnalyzer(analyzerClass)
          val stream = analyzer.tokenStream(key.toString, new StringReader(doc))
          val termAtt = stream.addAttribute(classOf[CharTermAttribute])
          stream.reset
          var buf = new ArrayBuffer[String]
          while(stream.incrementToken()) {
            if(termAtt.length > 0) {
              buf += new String(termAtt.buffer, 0, termAtt.length)
            }
          }
          stream.end
          stream.close
          (key, buf.toArray)
      }
    }

    def shingles(words: Array[String]): Array[String] = {
      val buf = new ListBuffer[Array[String]]()
      allowedNgrams.map(i => (Array.fill(i - 1)("_") ++ words ++ Array.fill(i - 1)("_")).sliding(i)) foreach buf.++=
      buf.result.map(_.mkString(" ")).toArray
    }

    def wordcounts[K:ClassTag](tokenized:RDD[(K, Array[String])]):RDD[(String, Int)] = {
      val wcs = if (allowedNgrams.end == 1) {
        tokenized.flatMap(_._2)
          .map { case (word) => (word, 1) }.reduceByKey(_ + _)
      } else {
        val allngrams =
          tokenized
            .map{case (doc, words) => shingles(words)}
            .flatMap {
              case (ngrams) =>
                ngrams.map(ngram =>
                    (ngram, ngram.substring(0, ngram.lastIndexOf(' ')), ngram.substring(ngram.lastIndexOf(' ') + 1)))
                  .flatMap { case (ng, ngh, ngt) => Array(((ngh, 'h'), ng), ((ngt, 't'), ng))}}
            .combineByKey((v: String) => Seq(v),
              (seq: Seq[String], v1: String) => seq ++ Seq(v1),
              (seq1: Seq[String],seq2: Seq[String]) => seq1 ++ seq2)
            .flatMap {
              case (sub, ngrams) =>
                val subgram = sub match {
                  case (head, 'h') => ((head, 'h'), ngrams.length)
                  case (tail, 't') => ((tail, 't'), ngrams.length / allowedNgrams.length)
                }
              ngrams.groupBy(ngram => ngram).toArray.map(group => ((group._1, group._2.length), subgram))
            }.groupByKey

        val ngramTotal = allngrams.count() / allowedNgrams.length
        val relevantngrams = allngrams.map { case (ngram, subgrams) =>
          val word = ngram._1
          val k11 = ngram._2
          val headntail = subgrams.toList
          val k12 = headntail(0)._2 - k11
          val k21 = headntail(1)._2 - k11
          val k22 = ngramTotal - (k12 + k21 + k11)
          val llr = LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22)
          (word, k11, llr)
        }.filter { case (word, k11, llr) => llr > minLLR && !word.contains("_") }
        relevantngrams.map { case (word, k11, llr) => (word, k11)}
      }
      wcs
    }

    val textRdd = sc.sequenceFile(path, classOf[IntWritable], classOf[Text], minPartitions = parMin)
      .map{ case (wKey, text) => wKey.get -> text.toString}

    val processIdf = if ("tfidf" equalsIgnoreCase weight) true else false
    val shouldPrune = maxDFSigma >= 0.0 || maxDFPercent < 100

    val tokenized = tokenize(textRdd)

    val (documentMatrix, dictionary) = {
      val wcsWithId = wordcounts(tokenized).zipWithIndex()
      val dict = wcsWithId.map { case (term_and_freq, id) => (term_and_freq._1, id.toInt) }
      val dictCollected = dict.collectAsMap()
      val numCols = dictCollected.map(_._2).max + 1
      val dictBcast = sc.broadcast(dictCollected)
      val tfVectors = tokenized
        .map {
          case (doc, tokens) =>
            val words:Array[String] = if(allowedNgrams.end>1) shingles(tokens) else tokens
            val bcast = dictBcast.value
            val vector:Vector = new RandomAccessSparseVector(numCols)
            words.groupBy(word => bcast.getOrElse(word, -1))
              .map{ case(i, arr)=> (i, arr.length.toDouble) }
              .filter(_._1 >= 0)
              .foreach{ e => vector.setQuick(e._1, e._2) }
          (doc, vector)
        }
      val tfMatrix = drmWrap(rdd = tfVectors, ncol = numCols, cacheHint = CacheHint.NONE)

      //do we need do calculate df-vector?
      if (shouldPrune || processIdf) {
        val numDocs = tfMatrix.nrow
        val dfVector = tfMatrix.numNonZeroElementsPerColumn()

        if (processIdf) {
          val idfVector = dfVector.cloned.assign(new DoubleFunction {
            override def apply(x: Double): Double = {
              if(x < minDF) 0 else{
                if(x/numDocs > maxDFPercent) 0 else Math.log(numDocs/x)
              }
            }
          })
          val idfVecBC = sc.broadcast(idfVector)
          val tfIdfMatrix = tfMatrix.mapBlock(ncol = tfMatrix.ncol){
            case (keys, block:Matrix) =>
              val idfVec = idfVecBC.value
              val res = block.like
              (0 until block.nrow).foreach{
                r => res(r, ::) := block.viewRow(r) * idfVec
              }
              (keys, res)

          }
          (tfIdfMatrix, dict.collectAsMap())

        } else {
          val pruner =  dfVector.cloned.assign(new DoubleFunction {
            override def apply(x: Double): Double = {
              if(x < minDF) 0 else{
                if(x/numDocs > maxDFPercent) 0 else 1
              }
            }
          })

          val prunerBC = sc.broadcast(pruner)
          val prunedDocumentMatrix = tfMatrix.mapBlock(ncol = tfMatrix.ncol){
            case (keys, block:Matrix) =>
              val pruneVec = prunerBC.value
              val res = block.like
              (0 until block.nrow).foreach{
                r => res(r, ::) := block.viewRow(r) * pruneVec
              }
              (keys, res)
          }
          (prunedDocumentMatrix, dict.collectAsMap())
        }
      } else {
        (tfMatrix, dict.collectAsMap())}
    }
    (documentMatrix, dictionary)
  }

  /** Parallelize in-core matrix as spark distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
                                  (implicit sc: DistributedContext)
  : CheckpointedDrm[Int] = {
    new CheckpointedDrmSpark(rdd = parallelizeInCore(m, numPartitions))
  }

  private[sparkbindings] def parallelizeInCore(m: Matrix, numPartitions: Int = 1)
                                              (implicit sc: DistributedContext): DrmRdd[Int] = {

    val p = (0 until m.nrow).map(i => i -> m(i, ::))
    sc.parallelize(p, numPartitions)

  }

  /** Parallelize in-core matrix as spark distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
                                 (implicit sc: DistributedContext)
  : CheckpointedDrm[String] = {

    val rb = m.getRowLabelBindings
    val p = for (i: String <- rb.keySet().toIndexedSeq) yield i -> m(rb(i), ::)

    new CheckpointedDrmSpark(rdd = sc.parallelize(p, numPartitions))
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
                         (implicit sc: DistributedContext): CheckpointedDrm[Int] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part => {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Int](rdd, nrow, ncol)
  }

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
                             (implicit sc: DistributedContext): CheckpointedDrm[Long] = {
    val rdd = sc.parallelize(0 to numPartitions, numPartitions).flatMap(part => {
      val partNRow = (nrow - 1) / numPartitions + 1
      val partStart = partNRow * part
      val partEnd = Math.min(partStart + partNRow, nrow)

      for (i <- partStart until partEnd) yield (i, new RandomAccessSparseVector(ncol): Vector)
    })
    new CheckpointedDrmSpark[Long](rdd, nrow, ncol)
  }

  private[mahout] def cacheHint2Spark(cacheHint: CacheHint.CacheHint): StorageLevel = cacheHint match {
    case CacheHint.NONE => StorageLevel.NONE
    case CacheHint.DISK_ONLY => StorageLevel.DISK_ONLY
    case CacheHint.DISK_ONLY_2 => StorageLevel.DISK_ONLY_2
    case CacheHint.MEMORY_ONLY => StorageLevel.MEMORY_ONLY
    case CacheHint.MEMORY_ONLY_2 => StorageLevel.MEMORY_ONLY_2
    case CacheHint.MEMORY_ONLY_SER => StorageLevel.MEMORY_ONLY_SER
    case CacheHint.MEMORY_ONLY_SER_2 => StorageLevel.MEMORY_ONLY_SER_2
    case CacheHint.MEMORY_AND_DISK => StorageLevel.MEMORY_AND_DISK
    case CacheHint.MEMORY_AND_DISK_2 => StorageLevel.MEMORY_AND_DISK_2
    case CacheHint.MEMORY_AND_DISK_SER => StorageLevel.MEMORY_AND_DISK_SER
    case CacheHint.MEMORY_AND_DISK_SER_2 => StorageLevel.MEMORY_AND_DISK_SER_2
  }

  /** Translate previously optimized physical plan */
  private def tr2phys[K: ClassTag](oper: DrmLike[K]): DrmRddInput[K] = {
    // I do explicit evidence propagation here since matching via case classes seems to be loosing
    // it and subsequently may cause something like DrmRddInput[Any] instead of [Int] or [String].
    // Hence you see explicit evidence attached to all recursive exec() calls.
    oper match {
      // If there are any such cases, they must go away in pass1. If they were not, then it wasn't
      // the A'A case but actual transposition intent which should be removed from consideration
      // (we cannot do actual flip for non-int-keyed arguments)
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAt(a) => At.at(op, tr2phys(a)(op.classTagA))
      case op@OpABt(a, b) => ABt.abt(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtB(a, b) => AtB.atb_nograph(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB),
        zippable = a.partitioningTag == b.partitioningTag)
      case op@OpAtA(a) => AtA.at_a(op, tr2phys(a)(op.classTagA))
      case op@OpAx(a, x) => Ax.ax_with_broadcast(op, tr2phys(a)(op.classTagA))
      case op@OpAtx(a, x) => Ax.atx_with_broadcast(op, tr2phys(a)(op.classTagA))
      case op@OpAewB(a, b, opId) => AewB.a_ew_b(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpCbind(a, b) => CbindAB.cbindAB_nograph(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpRbind(a, b) => RbindAB.rbindAB(op, tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAewScalar(a, s, _) => AewB.a_ew_scalar(op, tr2phys(a)(op.classTagA), s)
      case op@OpRowRange(a, _) => Slicing.rowRange(op, tr2phys(a)(op.classTagA))
      case op@OpTimesRightMatrix(a, _) => AinCoreB.rightMultiply(op, tr2phys(a)(op.classTagA))
      // Custom operators, we just execute them
      case blockOp: OpMapBlock[K, _] => MapBlock.exec(
        src = tr2phys(blockOp.A)(blockOp.classTagA),
        ncol = blockOp.ncol,
        bmf = blockOp.bmf
      )
      case op@OpPar(a,_,_) => Par.exec(op,tr2phys(a)(op.classTagA))
      case cp: CheckpointedDrm[K] => new DrmRddInput[K](rowWiseSrc = Some((cp.ncol, cp.rdd)))
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))

    }
  }

  /**
   * Returns an [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]] from default text
   * delimited files. Reads a vector per row.
   * @param src a comma separated list of URIs to read from
   * @param schema how the text file is formatted
   */
  def indexedDatasetDFSRead(src: String,
      schema: Schema = DefaultIndexedDatasetReadSchema,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create())
      (implicit sc: DistributedContext):
    IndexedDatasetSpark = {
    val reader = new TextDelimitedIndexedDatasetReader(schema)(sc)
    val ids = reader.readRowsFrom(src, existingRowIDs)
    ids
  }

  /**
   * Returns an [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]] from default text
   * delimited files. Reads an element per row.
   * @param src a comma separated list of URIs to read from
   * @param schema how the text file is formatted
   */
  def indexedDatasetDFSReadElements(src: String,
      schema: Schema = DefaultIndexedDatasetElementReadSchema,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create())
      (implicit sc: DistributedContext):
    IndexedDatasetSpark = {
    val reader = new TextDelimitedIndexedDatasetReader(schema)(sc)
    val ids = reader.readElementsFrom(src, existingRowIDs)
    ids
  }

}

