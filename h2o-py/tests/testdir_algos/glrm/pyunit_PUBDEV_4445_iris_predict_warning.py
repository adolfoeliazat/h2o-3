from __future__ import print_function
from builtins import str
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator


def glrm_iris():
  print("Importing iris_wheader.csv data...")
  irisH2O = h2o.upload_file(pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))
  irisTest = h2o.upload_file(pyunit_utils.locate("smalldata/iris/iris_wheader_bad_cnames.csv"))

  rank = 3
  gx = 0.5
  gy = 0.5
  trans="STANDARDIZE"
  print("H2O GLRM with rank k = " + str(rank) + ", gamma_x = " + str(gx) + ", gamma_y = " + str(gy) + ", transform = " + trans)
  glrm_h2o = H2OGeneralizedLowRankEstimator(k=rank, loss="Quadratic", gamma_x=gx, gamma_y=gy, transform=trans)
  glrm_h2o.train(x=irisH2O.names, training_frame=irisH2O)

  print("Impute original data from XY decomposition")
  pred_h2o = glrm_h2o.predict(irisTest)
  print("wow")



if __name__ == "__main__":
  pyunit_utils.standalone_test(glrm_iris)
else:
  glrm_iris()
