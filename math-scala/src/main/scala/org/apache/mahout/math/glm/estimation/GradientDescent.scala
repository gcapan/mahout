package org.apache.mahout.math.glm.estimation

import org.apache.mahout.logging._
import org.apache.mahout.math.drm.DrmLike

import org.apache.mahout.math.optimization._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{Matrices, Arrays, Vector, Matrix}

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
object GradientDescent{
  private val log = getLog(GradientDescent.getClass)

  def estimate[K: ClassTag](Xy: DrmLike[K], h: HypothesisFunction, l: LossFunction, classes:Option[Array[Int]],
                            randomSeed:Long = System.currentTimeMillis(), alpha:Double, lambda:Vector,
                            iterations: Int, r:Option[Double] = None): Matrix = {
    val nrows = classes match {
      case (Some(c)) =>
        if (c.length>2) c.length else 1
      case _ =>1
    }


    implicit val ctx = Xy.context
    val initial = drmBroadcast(Matrices.uniformView(nrows, Xy.ncol - 1, randomSeed.toInt))
    val lambdaVec = drmBroadcast(lambda)
    r match {
      case Some(d) => Xy.estimateWithBootstrapAveraging(d, localEstimator = (x, y) =>  gradientDescent(classes = None,
        initial.value, alpha, lambdaVec.value, iterations, h, l)(x, y))
      case _ => Xy.estimateWithAveraging(localEstimator = (x, y) =>  gradientDescent(classes = None, initial.value,
        alpha, lambdaVec.value, iterations, h, l)(x, y))
    }
  }

  def gradientDescent(classes: Option[Array[Int]], current: Matrix, alpha: Double, lambda: Vector,
                      iterations: Int, h: HypothesisFunction, l: LossFunction)
                     (X: Matrix, y: Vector): Matrix= {

    val cl = classes match {
      case (Some(c)) => classes
      case _ => if(current.nrow > 1) Some((0 until current.nrow).toArray) else classes
    }

    val yNew = cl match {
      case (Some(c)) =>
        //softmax
        if (c.length >2) {
          val ones = dense (Array.fill(c.length, y.size)(1.0) )
          (ones %*% diagv(y)) := ((i, j, v) => if (v == c(i) ) 1 else 0)
        }
        //logistic
        else dense(y.cloned := ((j, v) => if(v == c.last) 1 else 0))
      //other
      case _ => dense(y)
    }

    val paramUpdate = current.like
    val params = current.cloned

    val losses = Array.fill[Double](iterations)(0)
    var hOfX = h(X, params)

    for (k <- 0 until iterations) {
      for(i <- 0 until current.nrow) paramUpdate(i, ::) := gradient(hOfX, X, yNew, params(i, ::), lambda, i)

      params -= alpha * paramUpdate
      hOfX = h(X, params)
      losses(k) = l(hOfX, yNew)

      if(k > 0) {
        if (losses(k) > losses(k - 1)) {
          log.warn("Not converging")
        }
      }
      log.warn("In iteration %d, loss have been measured as %f".format(k+1, losses(k)))
    }
    params
  }

  def gradient(hOfX: Matrix, X:Matrix, y:Matrix, params: Vector, lambda:Vector, i:Int): Vector = {
    val ones = dense(dvec(Array.fill(X.ncol)(1.0)))
    ((dense(y(i, ::) - hOfX(::, i)).t %*% ones) * (- X)).colSums() + (lambda * params)
  }
}
