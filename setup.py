from setuptools import setup, find_packages
setup(
    name = 'touchgames',
    version = '0.1',

    author = 'Petr Viktorin',
    email = 'encukou@gmail.com',

    install_requires=[
        'PyMT >= 0.5',
        'Shapely >= 1.2',
        #'numpy >= 1.3',
    ],

    packages = find_packages(),
)
