package org.apache.mahout.math.scalabindings.elementwise

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.function.DoubleFunction
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._

import scala.reflect.ClassTag

/**
 * @author gokhan
 */


trait Unary[T]{
  def fn(f:Double=>Double): T
  def unary_- :T = fn(f = (a:Double) => -a)

}

object Unary{
  implicit def v2Unary(v:Vector) = new VectorUnary(v)
  implicit def m2Unary(m:Matrix) = new MatrixUnary(m)
  implicit def drm2Unary[K](drm:DrmLike[K])(implicit k:ClassTag[K]) = new DrmUnary(drm)

  def exp[T](data:Unary[T]) :T =  data.fn(f = (a:Double)=> Math.exp(a))
  def log[T](data:Unary[T]) :T = data.fn(f = (a:Double)=> Math.log(a))
  def log10[T](data:Unary[T]) :T = data.fn(f = (a:Double)=> Math.log10(a))
  def pow[T](data:Unary[T], to:Double) :T = data.fn(f = (a:Double)=>Math.pow(a, to))
}

class VectorUnary(self:Vector) extends Unary[Vector] {
  override def fn(f: (Double) => Double): Vector = self.cloned.assign(new DoubleFunction {
    override def apply(x: Double): Double = f(x)
  })
}
class MatrixUnary(self:Matrix) extends Unary[Matrix]{
  override def fn(f: (Double) => Double): Matrix = self.cloned.assign(new DoubleFunction {
    override def apply(x: Double): Double = f(x)
  })
}
class DrmUnary[K](self:DrmLike[K])(implicit k:ClassTag[K]) extends Unary[DrmLike[K]]{
  override def fn(f: (Double) => Double): DrmLike[K] = {
    import Unary._
    self.mapBlock(ncol = self.ncol, identicallyParitioned = true) {
      case (rowIds:Array[K], block:Matrix)=>(rowIds, block.fn(f))
    }
  }
}

