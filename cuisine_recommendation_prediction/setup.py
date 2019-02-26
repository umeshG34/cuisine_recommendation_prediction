from setuptools import setup, find_packages

setup(
    name='project2',
    version='1.0',
    author='Umesh Sai Gurram',
    author_email='umeshsai34@ou.edu',
    packages=find_packages(exclude=('tests', 'data')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
