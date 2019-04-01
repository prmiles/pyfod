# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import numpy as np
import unittest
from pyfod.GaussRiemannSum import GaussRiemannSum


# --------------------------
class Initialization(unittest.TestCase):

    def test_initialization(self):
        GRS = GaussRiemannSum(NGQ=10, start=1.0, finish=12.0)
        gpts = GRS.GQ.base_gauss_points()
        mlgpts = np.array([0.069431844202974, 0.330009478207572,
                           0.669990521792428, 0.930568155797026])
        self.assertEqual(gpts.size, 4, msg='Expect 4 elements')
        self.assertEqual(gpts.shape, (4,), msg='Expect shape = (4,)')
        self.assertTrue(np.allclose(gpts, mlgpts),
                        msg=str('Expect arrays to match: {} neq {}'.format(
                                gpts,
                                mlgpts)))



