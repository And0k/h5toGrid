import unittest
import pytest

from filters import *


# ##############################################################################
def test_find_sampling_frequency():
    n_insert = 30
    tim = np.array(['2017-10-20T12:36:31'] + ['2017-10-20T12:36:32'] * n_insert + ['2017-10-20T12:36:33'], '<M8[ns]')
    freq, n_same, n_decrease, i_inc = find_sampling_frequency(tim, precision=0.1, b_show=True)
    assert freq == pytest.approx(n_insert, abs=1e-6)
    assert n_same == n_insert
    assert n_decrease == 0
    assert np.alltrue(i_inc == np.int32([0, n_insert]))   # np.alltrue(np.diff(tim[i_inc]) > 0)
    pass


class LNDSTestCase(unittest.TestCase):
    # Try small lists and check that the correct subsequences are generated.
    def testLIS(self):
        # np.testing.assert_array_equal(
        #self.assertEqual(longest_increasing_subsequence_i(np.array([])), []) - NOT IMPLEMENTED because numba is very very bad
        self.assertEqual(list(longest_increasing_subsequence_i(np.arange(10, 0, -1))), [9])
        self.assertEqual(list(longest_increasing_subsequence_i(np.arange(10))), list(range(10)))
        self.assertEqual(list(longest_increasing_subsequence_i(
            np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]))), [3, 6, 9, 10, 13, 14])

    # def testLIS(self):
    #     self.assertEqual(other_filters.longest_increasing_subsequence_i(np.array([])),[])
    #     self.assertEqual(other_filters.longest_increasing_subsequence_i(np.arange(10,0,-1)),[1])
    #     self.assertEqual(other_filters.longest_increasing_subsequence_i(np.arange(10)),range(10))
    #     self.assertEqual(other_filters.longest_increasing_subsequence_i( \
    #         np.array([3,1,4,1,5,9,2,6,5,3,5,8,9,7,9])), [1, 2, 3, 5, 8, 9])
    #

    def test_repeated2increased(self):
        """ Test """
        # Function will auto increase frequency for some intervals if freq is low
        freq = 1
        t = np.array(['2017-10-20T12:36:32'] * 2 + ['2017-10-20T12:36:33'], '<M8[ns]')  # np.dtype('datetime64[ns]')
        t_out = repeated2increased(t.view(np.int64), freq, b_increased=None).view('<M8[ns]')
        self.assertTupleEqual(tuple(t_out), tuple(
            np.array(['2017-10-20T12:36:32', '2017-10-20T12:36:32.5', '2017-10-20T12:36:33'], '<M8[ns]')))

        # Increasing of frequency helps in most cases
        freq = 10
        t = np.hstack((np.datetime64('2017-10-20T12:36:32'),
                       np.arange(np.datetime64('2017-10-20T12:36:32'),
                                 np.datetime64('2017-10-20T12:36:35'),
                                 np.timedelta64(1, 's')), [
                       np.datetime64('2017-10-20T12:36:34')] * 3,
                       np.datetime64('2017-10-20T12:36:35'))
                      ).astype('<M8[ns]')
        t_out = repeated2increased(t.view(np.int64), freq, b_increased=None)
        self.assertTrue(np.all(np.diff(t_out) > 0))
        self.assertTupleEqual(tuple(t_out.view('<M8[ns]')), tuple(np.array(
            ['2017-10-20T12:36:32.0', '2017-10-20T12:36:32.1',
             '2017-10-20T12:36:33.0', '2017-10-20T12:36:34.0',
             '2017-10-20T12:36:34.1', '2017-10-20T12:36:34.2',
             '2017-10-20T12:36:34.3', '2017-10-20T12:36:35.0'],
            dtype='datetime64[ns]')))



if __name__ == '__main__':
    unittest.main()

"""
Running a single test module:
To run a single test module, in this case test_other_filters.py:

$ cd new_project
$ python -m unittest test.test_other_filters -v

Just reference the test module the same way you import it.

Running a single test case or test method:
Also you can run a single TestCase or a single test method:

$ python -m unittest test.test_other_filters.GravityTestCase
$ python -m unittest test.test_other_filters.GravityTestCase.test_method

Running all tests:
You can also use test discovery which will discover and run all the tests for you, they must be modules or packages named test*.py (can be changed with the -p, --pattern flag):

$ cd new_project
$ python -m unittest discover

This will run all the test*.py modules inside the test package.



	
python -m unittest discover
will find and run tests in the test directory if they are named test*.py. If you named 
the subdirectory tests, use
python -m unittest discover -s tests
, and if you named the test files other_filters_test.py, use
python -m unittest discover -s tests -p '*test.py'
File names can use underscores but not dashes.

"""
