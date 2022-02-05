# -*- coding: utf-8 -*-
import unittest
import concurrent.futures
import os

import numpy as np
import numba

import fractalshades as fs
import fractalshades.parallel_lock
import test_config

"""
A few Experiments with numba & GIL releasing in multi-threading.
"""

# @numba.njit("(intp[:], float64[:])", nogil=True)
@numba.njit(nogil=True)
def lock_and_work(locks, arr):
    failtimes = 0
    for _ in range(10000):
        for i in range(locks.size):
            # get lock pointer
            lock_ptr = locks[i:]
            # try to lock and do some work
            if fs.parallel_lock.try_lock(lock_ptr):
                arr[i] += 1
                # unlock
                fs.parallel_lock.unlock(lock_ptr)
                break
        else:
            # count number of times it failed to do work
            failtimes += 1
    return failtimes


@numba.njit(nogil=True)
def busywait_and_work(locks, arr):
    failtimes = 0
    for _ in range(10000):
        for i in range(0):#locks.size):
            # get lock pointer
            lock_ptr = locks[i:]
            # busywait to lock and do some work
            while not(fs.parallel_lock.try_lock(lock_ptr)):
                pass
            arr[i] += 1
            fs.parallel_lock.unlock(lock_ptr)
            break
        else:
            # count number of times it failed to do work
            failtimes += 1
    return failtimes

@numba.njit(nogil=False)
def lock_and_work_with_GIL(locks, arr):
    failtimes = 0
    for _ in range(10000):
        for i in range(locks.size):
            # get lock pointer
            lock_ptr = locks[i:]
            # try to lock and do some work
            if fs.parallel_lock.try_lock(lock_ptr):
                arr[i] += 1
                # unlock
                fs.parallel_lock.unlock(lock_ptr)
                break
        else:
            # count number of times it failed to do work
            failtimes += 1
    return failtimes

@numba.njit(nogil=True)
def lock_and_work_nested(locks, arr):
    return lock_and_work_with_GIL(locks, arr)


def python_lock_and_work(locks, arr):
    # just passing through
    return lock_and_work(locks, arr)

def python_lock_and_work_nested(locks, arr):
    # just passing through
    return lock_and_work_nested(locks, arr)


class Test_parallel(unittest.TestCase):
    
    def test_lock(self):
        # calling a numba nogil=True jitted func
        # -> GIL is released (as expected)
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(lock_and_work, locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert np.sum(values) + lock_failed == 90000.

    def test_lock_python(self):
        # calling a numba nogil=True jitted func through a python func
        # -> GIL is released
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(python_lock_and_work, locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert np.sum(values) + lock_failed == 90000.

    def test_lock_GIL(self):
        # Calling a numba function compiled with nogil=False
        # -> GIL is not released (as expected)
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(lock_and_work_with_GIL, locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert lock_failed == 0.
        assert np.sum(values) == 90000.

    def test_lock_nested(self):
        # Calling nested :
        # - a nogil=True numba jitted function
        # - which calls a nogil=False numba jitted
        # -> GIL is released
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(lock_and_work_nested, locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert np.sum(values) + lock_failed == 90000.

    def test_lock_python_nested(self):
        # Calling nested :
        # - pure python function
        # - which calls a nogil=True numba jitted
        # - which calls a nogil=False numba jitted
        # -> GIL is released
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(python_lock_and_work_nested,
                                         locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert np.sum(values) + lock_failed == 90000.

    def test_lock2(self):
        # Syntax for in place mod 
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (
                threadpool.submit(lock_and_work, locks, values)
                for _ in range(8)
            )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
        print(np.sum(values))

    def test_busy_lock(self):
        # calling a numba nogil=True jitted func
        # -> GIL is released (as expected)
        locks = np.zeros(3, dtype=np.intp)
        values = np.zeros(3, dtype=np.float64)
        assert lock_and_work(locks, values) == 0
        assert np.sum(values) == 10000.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()
                    ) as threadpool:
            futures = (threadpool.submit(busywait_and_work, locks, values) 
                       for _ in range(8))
            lock_failed = 0
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                lock_failed += res
                # print('failed to lock {0} times'.format(res))
        print('total failed to lock:', lock_failed)
        print(np.sum(values))
        assert np.sum(values) + lock_failed == 90000. 
        
        
    
if __name__ == '__main__':
    # main()
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_parallel]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_parallel("test_busy_lock"))
        runner.run(suite)

