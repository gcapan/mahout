package org.apache.mahout.math.optimization

import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
 * @author gokhan
 */
trait DistributedOptimizationSuiteBase extends DistributedMahoutSuite with Matchers{
  this: FunSuite =>

  
  test("averaging") {
    val inCoreXy = dense((3, 4, 1.0), (5, 6, 0.0), (7, 8, 1.0))
    val Xy = drmParallelize(m = inCoreXy, numPartitions = 2)

    val ops = new OptimizationDistributionOps(Xy)
    val res = ops.estimateWithAveraging(localEstimator = (X, y) => dense(dvec(1, 2)))
    res.ncol should equal(2)
    res.nrow should equal(1)
    res(0, ::) should equal(dvec(1, 2))
  }

  test("bootstrap averaging") {
    val inCoreXy = dense((3, 4, 1.0), (5, 6, 0.0), (7, 8, 1.0))
    val Xy = drmParallelize(m = inCoreXy, numPartitions = 2)
    val r = 0.7

    val ops = new OptimizationDistributionOps(Xy)
    val res = ops.estimateWithBootstrapAveraging(r, localEstimator = (X, y) => dense(dvec(1, 2)))
    res.ncol should equal(2)
    res.nrow should equal(1)
    res(0, ::) should equal((dvec(1, 2) - r * dvec(1, 2)) / (1 - r))

  }

}
