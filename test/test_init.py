# -*- coding: utf-8 -*-

import unittest
import pyfod


class ImportPyfod(unittest.TestCase):

    def test_version_attribute(self):
        version = pyfod.__version__
        self.assertTrue(isinstance(version, str),
                        msg='Expect string output')
        self.assertEqual(len(version.split('.')), 3,
                         msg='Expect #.#.# format')