import unittest

class TestExample(unittest.TestCase):
    def test_failing(self):
        self.assertTrue(False, "This test should fail")
    
    def test_another_failing(self):
        self.assertEqual(1, 2, "Math is broken")

if __name__ == '__main__':
    unittest.main(verbosity=2)
