package org.apache.mahout.math.optimization

import org.apache.mahout.logging._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.Vector
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._


import scala.reflect.ClassTag
import scala.util.Random

/**
 * @param Xy X:y, where y (last column of X) represents the target variable
 */
class OptimizationDistributionOps[K: ClassTag](val Xy: DrmLike[K]) {
  private val log = getLog(getClass)

  def estimateWithAveraging(localEstimator: (Matrix, Vector)=>Matrix):Matrix = {
    val m = Xy.allreduceBlock(
      bmf = {case (indices, loc) => localEstimator(loc(::, 0 until loc.ncol-1), loc(::, loc.ncol-1)).cbind(1)},
      rf = (m1, m2) => m1 + m2)
    m(::, 0 until m.ncol-1) / m(0, m.ncol - 1)

  }
  def estimateWithBootstrapAveraging(r:Double, localEstimator: (Matrix, Vector)=>Matrix):Matrix = {
    val m = Xy.allreduceBlock(
      bmf = {case (indices, loc) => 
        val rows = Random.shuffle((0 until loc.nrow).toList).take((r * loc.nrow).ceil.toInt)
        localEstimator(loc(::, 0 until loc.ncol-1), loc(::, loc.ncol-1)).cbind(1).
        rbind(localEstimator(loc(rows)(::, 0 until loc.ncol-1), loc(rows)(::, loc.ncol-1)).cbind(1))
      },
      rf = (m1, m2) => m1 + m2
    )
    val allAvg = m(::, 0 until m.ncol - 1) / m(0, m.ncol - 1)
    (allAvg(0 until allAvg.nrow/2, ::) - allAvg(allAvg.nrow/2 until allAvg.nrow, ::) * r) / (1 - r)

  }

}
