# -*- coding: utf-8 -*-
import unittest
import fractalshades.utils as fsutils
import test_config

class Test_store_kwargs(unittest.TestCase):

    def test_method_with_all_kwargs(self):
        class A:
            @fsutils._store_kwargs("my_options")
            def foo(self, *, x, y, z):
                pass
        a = A()
        kwargs = {"x": 1, "y": 2, "z": 3}
        a.foo(**kwargs)
        for k, v in kwargs.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.my_options, kwargs)
        
    def test_method_with_default_kwargs(self):
        class A:
            @fsutils._store_kwargs("my_options")
            def foo(self, *, x=10, y=20, z=30):
                pass
        a = A()
        kwargs = {"x": 1, "y": 2}
        expected = {"x": 1, "y": 2, "z": 30}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.my_options, expected)

    def test_call_dict_persistent(self):
        class A:
            @fsutils._store_kwargs("my_options")
            def foo(self, *, x, y, z=[3]):
                pass
        a = A()
        kwargs = {"x": [1], "y": 2}
        expected = {"x": [1], "y": 2, "z": [3]}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.my_options, expected)
        a.x[0] = -1
        a.z[0] = -1
        self.assertEqual(a.my_options, expected)

    def test_method_with_args(self):
        class A:
            @fsutils._store_kwargs("my_options")
            def foo(self, x, y, z=[3]):
                pass
        a = A()
        kwargs = {"x": [1], "y": 2}
        expected = {"x": [1], "y": 2, "z": [3]}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.my_options, expected)
        self.assertRaises(TypeError, a.foo, ([1], 2))

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
        self.assertEqual(a.zoom_options, expected)
        # Test with a default value
        kwargs = {"x": [-1], "y": -2}
        expected = {"x": [-1], "y": -2, "z": [3]}
        a.foo(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.zoom_options, expected)

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
            @fsutils.zoom_options
            def foo(self, x, y, z=[3]):
                pass
            def foo2(self, x, y, z=[3]):
                pass

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
        expected = {"x": [1], "y": 2, "z": 4}
        a.calc1(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.calc_options, expected)
        self.assertEqual(a.calc_options_callable, "calc1")
#        # Test with a default value
        kwargs = {"x": [-1], "y": -2}
        expected = {"x": [-1], "y": -2, "z": [30]}
        a.calc2(**kwargs)
        for k, v in expected.items():
            self.assertEqual(getattr(a, k), v)
        self.assertEqual(a.calc_options, expected)
        self.assertEqual(a.calc_options_callable, "calc2")

if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_store_kwargs]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_store_kwargs("test_method_with_all_kwargs"))
        runner.run(suite)
