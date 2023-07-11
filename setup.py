import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='hierarchical_qp',
    version='0.0.1',
    author='Davide De Benedittis',
    author_email='davide.debenedittis@gmail.com',
    description='Hierarchical Quadratic Programming implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ddebenedittis/hierarchical_qp',
    project_urls = {
        "Bug Tracker": "https://github.com/ddebenedittis/hierarchical_qp/issues"
    },
    license='MIT',
    packages=['hierarchical_qp'],
    install_requires=['numpy', 'quadprog'],
)