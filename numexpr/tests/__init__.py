def test():
    import nose.core
    nose.core.main()
test.__test__ = False   # tell nose this is not a test

if __name__ == '__main__':
    test()
