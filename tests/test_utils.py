# -*- coding: utf-8 -*-
import unittest
import fractalshades.utils as fsutils
import test_config

class Test_store_kwargs(unittest.TestCase):


    def test_zoom_decorator(self):
        class A:
            @fsutils.zoom_options
            def foo(self, x, y, z=[3]):
                pass
        a = A()
        # Test with a all values
        kwargs = {"x": [1], "y": 2, "z": 4}
        expected = {"x": [1], "y": 2, "z": 4}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
            self.assertEqual(a.zoom_kwargs[k], v)

        # Test with a default value
        kwargs = {"x": [-1], "y": -2}
        expected = {"x": [-1], "y": -2, "z": [3]}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
            self.assertEqual(a.zoom_kwargs[k], v)


    def test_calc_decorator(self):
        class A:
            @fsutils.calc_options
            def calc1(self, x, y, z=[3]):
                pass
            @fsutils.calc_options
            def calc2(self, x, y, z=[30]):
                pass
            @fsutils.calc_options
            def calc3(self, x, y, z=[300]):
                pass
            @fsutils.interactive_options
            def foo(self, x, y, z=[3]):
                pass
            def foo2(self, x, y, z=[3]):
                pass
            @fsutils.interactive_options
            def foo3(self, x, y, z=[3]):
                pass
            def calc_hook(self, calc_callable, calc_kwargs, return_dic):
                setattr(self, calc_callable + "__kwargs", calc_kwargs)

        # Test that the method decorated with calc_options are recognized
        # Note that it is not the bound method but the underlying func
        # (class arribute, not instance attribute)
        tagged = {"calc1": A.calc1,
                  "calc2": A.calc2,
                  "calc3": A.calc3}
        self.assertDictEqual(fsutils.calc_options.methods(A), tagged)

        a = A()
        # Test with a all values
        kwargs = {"x": [1], "y": 2, "z": 4}
        a.calc1(**kwargs)
        self.assertEqual(a.calc1__kwargs, kwargs)

        # Test with a default value
        kwargs = {"x": [-1], "y": -2}
        expected = {"x": [-1], "y": -2, "z": [30]}
        a.calc2(**kwargs)
        self.assertEqual(a.calc2__kwargs, expected)
        
        # Test GUI listing for interactive_options
        GUI_dic = fsutils.interactive_options.methods(A)
        self.assertEqual(set(GUI_dic.keys()), set(("foo", "foo3")))
        



if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_store_kwargs]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_store_kwargs("test_calc_decorator"))
        runner.run(suite)
