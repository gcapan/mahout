package org.apache.mahout.math

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.math._
import scala.reflect.ClassTag


/**
 * @author gokhan
 */
package object optimization {
  /**
   * given predicted y, actual y, X[i,:], and the parameters vector,
   * what is the gradient that would be element-wise subtracted from the current parameters vector?
   * Example - gradient for L2-regularized squared loss minimization:
   * (hOfx, y, x, b) => x * (hOfx - y)
   */
  type GradientFunction = (Double, Double, Vector, Vector) => Vector

  /**
   * The hypothesis function given x dot b
   * Example - sigmoid function:
   * (x, theta) => 1/ 1 + Math.exp(-xDotb)
   */
  type HypothesisFunction = Double => Double

  /**
   * Error amount given X (mxn), b(nx1), Y(mx1)
   */
  type ErrorMeasure[K] = (DrmLike[K], Vector, DrmLike[K]) => Double

  val defaultGradient = (hOfx: Double, y: Double, x: Vector, b: Vector) => -x * (y - hOfx)
  val defaultHypothesis = (xDotb: Double) => xDotb

  /**
   * the sigmoid function
   * x instance
   * b parameters
   */
  //val logisticHypothesis = (xDotb: Double) => 1 / (1 + exp(- xDotb))


  def rmse[K: ClassTag] = (X: DrmLike[K], b: Vector, y: DrmLike[K]) => {
    (X %*% b - y).norm / sqrt(y.ncol * y.nrow)
  }
}
