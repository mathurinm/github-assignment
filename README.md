# Github Assignment

Forked from https://github.com/x-datascience-datacamp/datacamp-assignment-numpy, project started by Thomas Moreau and Alexandre Gramfort.

## What we want you to learn by doing this assignment:

  - Use Git and GitHub
  - Work with Python files (and not just notebooks!)
  - Do a pull request on a GitHub repository
  - Format your code properly using standard Python conventions
  - Make your code pass tests run automatically on a continuous integration system (GitHub actions)

## How?

  - Fork the repository by clicking on the `Fork` button on the upper right corner
  - Clone the repository of your fork with: `git clone https://github.com/MYLOGIN/github-assignment` (replace MYLOGIN with your GitHub login)
  - Create a branch called `myname` using `git checkout -b myname`.
  - Add an X at the end of the line with your name in the file `students.txt`.
  - Push your branch to your repository.
  - Send a Pull Request of this branch to this repository (mathurinm/github-assignment; not your fork).
  - Ping me on Slack so that I merge the PR. This will allow the test suite (COnitnuous Integration) to be run on your future PRs.
  - Go back to your `main` branch with `git checkout main`.
  - Create a branch called `myassignment` using `git checkout -b myassignment`
  - Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.
  - Open the pull request on GitHub
  - Keep pushing to your branch until the continuous integration (CI) system is green. Alternatively, you can run the tests locally with the following command:
  ```
    flake8 .
    pydocstyle
  ```
  - When it is green notify the professors that you are done.
