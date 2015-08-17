package org.apache.mahout.math.glm

import org.apache.mahout.math.{Matrices, Matrix}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math._

/**
 * @author gokhan
 */
package object estimation {



  /*h(X, params)*/
  type HypothesisFunction = (Matrix, Matrix) => Matrix

  /*l(h(X, params), y)*/
  type LossFunction = (Matrix, Matrix) => Double


  val linearRegressionHypothesis: HypothesisFunction = {
    case (x, params) => dense(x %*% params(0, ::)).t
  }
  val linearRegressionLoss: LossFunction = {
    case (h, y) => ((y(0, ::) - h(::, 0))^2).sum * 0.5
  }

  val logisticRegressionHypothesis: HypothesisFunction = {
    case(x, theta) => dense(1 / (1 + vexp(-(x %*% theta(0, ::))))).t
  }
  val logisticRegressionLogLoss: LossFunction = {
    case(h, y) => ((y(0, ::) * vlog(h(::, 0))) + ((1 - y(0, ::)) * vlog(1 - h(::, 0)))).sum * -1
  }


  val softmaxHypothesis: HypothesisFunction = {
    case (x, theta) =>
      val tmp = mexp(x %*% theta.t)
      diagv(1 / tmp.rowSums) %*% tmp
  }

  val softmaxLogLoss: LossFunction = {
    case (h, y) =>
      (y.t * mlog(h)).rowSums.sum * -1
  }

}
