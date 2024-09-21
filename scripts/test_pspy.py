import sys


def main():
    from pspy.tests.test_so_spectra import SOSpectraTests

    test = SOSpectraTests()

    # verbose = "-v" in sys.argv
    verbose = True

    test.setUp(verbose=verbose)
    test.test_spectra(verbose=verbose)


if __name__ == "__main__":
    main()
