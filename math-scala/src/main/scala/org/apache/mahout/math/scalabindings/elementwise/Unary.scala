package org.apache.mahout.math.scalabindings.elementwise

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._

/**
 * @author gokhan
 */


abstract class Unary[T] (self:T){
  def fn(f:Double=>Double): T
  class Ops(a: T) {
    def unary_- :T = fn(f = (a:Double) => -a)
    def exp:T = 
  }
}
object ElementWiseOps{
  trait Implicits {
    implicit def mathOps[T](x: T)(implicit ew: Unary[T])
  }
}

class VectorUnary(self:Vector) extends Unary[Vector](self:Vector) {
  override def fn(f: (Double) => Double): Vector = self.cloned.assign(f)
}
class MatrixUnary(self:Matrix) extends Unary[Matrix](self:Matrix) {
  override def fn(f: (Double) => Double): Matrix = self.cloned.assign(f)
}
class DrmUnary[K](self:DrmLike[K]) extends Unary[DrmLike[K]](self:DrmLike[K]) {
  override def fn(f: (Double) => Double): DrmLike[K] = self.mapBlock(ncol = self.ncol,
    identicallyParitioned = true, bmf = {
      case (rowIds, block) => (rowIds, new MatrixUnary(block).apply(f))
    })
}

