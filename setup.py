#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()  # ['Click>=7.0', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="daniele de gregorio",
    author_email='daniele.degregorio@eyecan.ai',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python library for deep learning data manipulation",
    entry_points={
        'console_scripts': [
            'persefone=persefone.cli:main',
            'persefone_database_f2d=persefone.cli.tools.converters.folder_to_mongo:folder_to_mongo',
            'persefone_database_d2f=persefone.cli.tools.converters.mongo_to_folder:mongo_to_folder',
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='persefone',
    name='persefone',
    packages=find_packages(include=['persefone', 'persefone.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/m4nh/persefone',
    version='0.0.9',
    zip_safe=False,
)
