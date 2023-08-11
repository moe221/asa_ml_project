from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='cpi_targets_ml',
      version="0.0.0",
      description="CPI target model",
      license="MIT",
      author="Mohamed Abuhalala",
      author_email="mohamed.abuhalala@phiture.com",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
