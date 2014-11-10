package org.apache.mahout.math.optimization

import org.apache.log4j.Logger
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.Random

/**
 * @author gokhan
 */
private[math] object SGD {
  private val log = Logger.getLogger(SGD.getClass)

  class Result(val b: Vector, val iterationsError: Iterable[Double]){
    def toTuple = (b, iterationsError)
  }

  /**
   *
   * @param XandY (Possibly a block of) data and y (X.cbind(y)), but this can be used for inCore SGD, too
   * @param b parameters
   * @param gradient given predicted y, actual y, X[i,:], and the parameters vector,
   *                 what is the gradient that would be element-wise subtracted from the current parameters vector?
   *                 Example - gradient for L2-regularized squared loss minimization (lambda term handled
   *                 automatically):
   *                 (hOfx, y, x, b) => -x * (y - hOfx(x dot b))
   * @param hOfx     The hypothesis function given x dot b
   * @param alpha Learning rate
   * @param lambda Regularization rate
   */
  private def minimizeInCoreWithSgd (XandY: Matrix, b: Vector,
      gradient: GradientFunction,
      hOfx: HypothesisFunction,
      alpha: Double, lambda: Double) = {

    val last = XandY.ncol-1
    val y = XandY(::, last)
    val X = XandY(::, 0 until last-1)
    var bCloned = b.cloned

    Random.shuffle(X.iterator().asScala).foreach(slice => {
      val i = slice.index()
      val x_i = slice.vector()
      bCloned -= alpha * (gradient(hOfx(x_i dot bCloned), y.get(i), x_i, bCloned) + lambda * bCloned)
    })

    bCloned
  }

  /**
   * Parallel stochastic gradient descent based on Zinkevich, Martin A. et al's "Parallelized Stochastic Gradient
   * Descent", NIPS 2010 paper
   * http://martin.zinkevich.org/publications/nips2010.pdf
   * 
   * @param X data mxn
   * @param y actual y vector, mx1
   * @param maxIterations max number of iterations for SGD
   * @param convergenceThreshold whether the algorithm is converged
   * @param alpha learning rate
   * @param lambda (L2 penalty) regularization rate, 0 (no regularization) by default
   * @param hOfx function that calculates the corresponding y_i from X[i, :]xb
   * @param gradient given predicted y, actual y, X[i,:], and the parameters vector,
   *                 what is the gradient that would be element-wise subtracted from the current parameters vector
   * @return
   */
  def minimizeWithSgd[K: ClassTag](
      X:DrmLike[K],
      y:DrmLike[K],
      maxIterations:Int = 10,
      convergenceThreshold: Double = 0.10,
      alpha: Double,
      lambda: Double = 0.0,
      hOfx: HypothesisFunction,
      gradient: GradientFunction,
      error: ErrorMeasure[K]): Result = {

    var b:Vector = Matrices.symmetricUniformView(1, X.ncol, RandomUtils.getRandom.nextInt).viewRow(0).cloned
    var errorPerIteration: List[Double] = Nil

    var stop = false
    var i = 0

    /*TODO: I need to replace the mapBlock call below, with a higher order function to which I can pass a function
     *that would return a vector with size n from a matrix with size mxn*/

    while (!stop && i < maxIterations) {
      val minimizePartial = SGD.minimizeInCoreWithSgd(_:Matrix, b, gradient, hOfx, alpha, lambda)
      //mapBlock call is just for illustration purposes
      val bMatrix = X.cbind(y).mapBlock() {
        case(keys, block) =>
          minimizePartial(block)
          //For this to compile
          (keys, block)
      }.collect
        //Get the sum of all b's independently calculated in different blocks and divide it by number of such b's:
        // when the matrix is comprised of rows of b vectors, colSums is equivalent to the sum of all b's
      b = bMatrix.colSums() /= bMatrix.nrow


      if (convergenceThreshold > 0) {
        val err = error(X, b, y)

        if (i > 0) {
          val errPrev = errorPerIteration.last
          val convergence = (errPrev - err) / errPrev

          if (convergence < 0) {
            log.warn("RMSE increase of %f".format(convergence))
            stop = true
          } else if (convergence < convergenceThreshold) {
            stop = true
          }
        }
        errorPerIteration :+= err
      }
      i += 1
    }

    //Dummy return value
    new Result(b, Array[Double](maxIterations))
  }
}
