
# Contributors Guide

1. Open an issue indicating what you would like to do
2. Clone the repo at <https://github.com/jfcrenshaw/photerr>
3. Install [poetry](https://python-poetry.org/docs/#installation)
4. From inside the `photerr` directory, run `poetry install`
5. Create a new branch of the repository and begin development.
6. Submit a pull request. There are a number of things you should check:
    - Make sure that you format your code according to flake8, isort, and black (these install with the development dependencies and should automatically use the correct settings described in `pyproject.toml`)
    - Before merging, rebase your branch and squash all of your commits into one or two clear commits. This is to keep the version history clean and linear.
    - Make sure you bump the *semantic* version number.
7. After merging your changes, create a new release on Github. Make sure you include a description of the changes, as I use the release page to track changes.
