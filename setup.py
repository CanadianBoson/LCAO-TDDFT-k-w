from distutils.core import setup

setup(name='lcaotddft-k-omega',
      version='0.1',
      description='LCAO TDDFT for GPAW in frequency space',
      maintainer='Duncan John Mowbray',
      maintainer_email='duncan.mowbray@gmail.com',
      url='https://gitlab.com/lcao-tddft-k-omega/lcao-tddft-k-omega',
      platforms=['unix'],
      license='GPLv3+',
      scripts=['bin/lcao-tddft-k-omega',
               'bin/exciton-density',
               'bin/density-difference'],
      packages=['lcaotddftkomega'])
