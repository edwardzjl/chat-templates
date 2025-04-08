import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module in ["qwen2_5"]:
        suite.addTests(loader.discover(f"{module}/tests"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
