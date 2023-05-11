from setuptools import setup

setup(
   name='sphpc',
   version='0.1.0',
   author='ddrous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['sphpc', 'sphpc.tests'],
#    scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/sphpc/',
   license='LICENSE.md',
   description='AI-enabled simulator and controller for high performance smoothed particles hydrodynamics',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
       "jax >= 0.3.4",
       "pytest",
       "matplotlib>=3.4.0",
   ],
)
