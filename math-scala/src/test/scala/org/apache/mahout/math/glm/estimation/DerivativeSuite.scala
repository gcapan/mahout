package org.apache.mahout.math.glm.estimation

import org.apache.mahout.logging._
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math._


/**
 * @author gokhan
 */
class DerivativeSuite extends FunSuite with MahoutSuite{
  private final implicit val log = getLog(classOf[DerivativeSuite])

  private def approximate(x: Matrix, y: Matrix, theta:Matrix, lambda: Vector, i:Int, j:Int, h:HypothesisFunction,
                          l: LossFunction): Double = {
    val diff = 1e-5
    val thetaClose = theta.cloned
    thetaClose(i, j) += diff

    (l(h(x, thetaClose), y) + ((thetaClose %*% diagv(lambda / 2))^2).sum -
      (l(h(x, theta), y) + ((theta %*% diagv(lambda / 2))^2).sum)) / diff
  }


  private def compareWithApprox(x: Matrix, y: Matrix, theta: Matrix, lambda:Vector, h:HypothesisFunction,
                                l: LossFunction) {
    val gradient = theta.like
    for(i <- 0 until y.nrow) {
      gradient(i, ::) = GradientDescent.gradient(h(x, theta), x, y, theta(i, ::), lambda, i)
    }

    for (i <- 0 until theta.nrow){
      for (j <- 0 until theta.ncol) {
        val approx = approximate(x, y, theta, lambda, i, j, h, l)
        println("Err: %d %d %f".format(i, j, gradient(i, j) - approx))

        (gradient(i, j) * approx) should be > 0.0
        (gradient(i, j) - approx) should be < 1e-3
      }
    }
  }

  test("gradient computation, ridge regression") {
    val h: HypothesisFunction = linearRegressionHypothesis
    val l: LossFunction= linearRegressionLoss
    val theta = dense(dvec(0.1, 0.1, 0.1))
    val lambda = dvec(0.01, 0.01, 0.01)
    val x = dense(dvec(1, 0.2, 0.3), dvec(1, 0.4, 0.5))
    val y = dense(dvec(10, 20))
    compareWithApprox(x, y, theta, lambda, h, l)
  }

  test("gradient computation, L2 regularized logistic regression"){
    val h: HypothesisFunction = logisticRegressionHypothesis
    val l: LossFunction = logisticRegressionLogLoss
    val theta = dense(dvec(0.1, 0.1, 0.1))
    val lambda = dvec(0.01, 0.01, 0.01)
    val x = dense(dvec(1, 0.2, 0.3), dvec(1, 0.4, 0.5))
    val y = dense(dvec(0, 1))
    compareWithApprox(x, y, theta, lambda, h, l)
  }

  test("gradient computation, L2 regularized softmax regression"){
    val h: HypothesisFunction = softmaxHypothesis
    val l: LossFunction = softmaxLogLoss
    val theta = dense(dvec(0.1, 0.1, 0.1), dvec(0.1, 0.1, 0.1), dvec(0.1, 0.1, 0.1))
    val lambda = dvec(0.01, 0.01, 0.01)
    val x = dense(dvec(1, 0.2, 0.3), dvec(1, 0.4, 0.5), dvec(1, 0.1, 0.8), dvec(1, 0.6, 0.7))
    val y = dense(dvec(1, 0, 0), dvec(0, 1, 0), dvec(0, 0, 1), dvec(0, 1, 0)).t
    compareWithApprox(x, y, theta, lambda, h, l)

  }

}
