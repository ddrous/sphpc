from setuptools import setup

setup(
   name='Scale',
   version='0.1.0',
   author='desmondous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['scale', 'scale.tests'],
#    scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/Scale/',
   license='LICENSE.md',
   description='Simulator and controller for adjacency-based large equations',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
       "jax >= 0.3.4",
       "pytest",
       "matplotlib>=3.4.0",
   ],
)
