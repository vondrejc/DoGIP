#!/usr/bin/python

import unittest


if __name__ == "__main__":
    from dogip.unittests import Test_FEMsimplex
    from examples.FEMsimplex.weighted_projection import Test_weighted_projection
    from examples.FEMsimplex.elliptic_problem import Test_elliptic_problem

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_FEMsimplex))
    suite.addTest(unittest.makeSuite(Test_weighted_projection))
    suite.addTest(unittest.makeSuite(Test_elliptic_problem))

    runner=unittest.TextTestRunner()
    runner.run(suite)
