package org.apache.mahout.math

import org.apache.mahout.math.drm.DrmLike

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
package object optimization {
  implicit def drm2OptimizationOps[K: ClassTag](cp: DrmLike[K]) = new OptimizationDistributionOps(cp)
}
