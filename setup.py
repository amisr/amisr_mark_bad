from setuptools import setup, find_packages

from amisr_mark_bad import __version__

setup(
    name='amisr_mark_bad',
    version=__version__,

    url='https://github.com/pmreyes2/amisr_mark_bad',
    author='Pablo Reyes',
    author_email='pablo.reyes@sri.com',

    #packages=find_packages(where="amisr_mark_bad"), # this only searches inside amisr_mark_bad
    packages=find_packages(),#exclude=["mark_data.py"]),
    install_requires=[
            'bokeh','h5py','numpy'
            ],
    entry_points={  # Optional
        'console_scripts': [
            'amisr_mark_bad=amisr_mark_bad.bin.__init__:main',
            ],
        },
)
