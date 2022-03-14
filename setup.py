from setuptools import find_packages, setup

setup(
    name='EntFA',
    version='1.0',
    description="Check our ACL 2022 paper 'Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization'",
    author='Meng Cao',
    author_email='meng.cao@mail.mcgill.ca',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[],  # external packages as dependencies
)
