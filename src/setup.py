from distutils.core import setup

setup(
    name='easy_ica',
    version='0.1',
    packages=[''],
    url='https://github.com/AbelHLG/easy_ica',
    license='GPLv2',
    author='Abel Antonio Fernandez Higuera,'
           'Roberto Antonio Becerra Garcia, '
           'Rodolfo Valentin Garcia Bermudez',
    author_email='afernandezh@facinf.uho.edu.cu',
    description='Independent component analysis (ICA) using maximum likelihood, square mixing matrix and no noise '
                '(Infomax).Source prior is assumed to be p(s)=1/pi*exp(-ln(cosh(s))). For optimization the BFGS '
                'algorithm is used.',
    py_modules=['ica'],
)
