"""Tests for `pspy.so_dict` object."""

import tempfile
import unittest

from pspy import so_dict


class SODictTests(unittest.TestCase):
    def test_getter(self):
        d = so_dict.so_dict()
        with self.assertRaises(ValueError):
            _ = d["foo"]
        with self.assertRaises(ValueError):
            _ = d.get("foo", raise_error=True)

        self.assertEqual(d.get("foo", 666), 666)

    def test_read_from_file(self):
        s = b"""
        foo = 666
        bar = "bar"
        """
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(s)
            tmp.seek(0)
            d = so_dict.so_dict()
            d.read_from_file(tmp.name)
            self.assertTrue(d.get("foo") != None)
            self.assertTrue(d.get("bar") == "bar")


if __name__ == "__main__":
    unittest.main()
