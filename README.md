# Github Assignment

Forked from https://github.com/x-datascience-datacamp/datacamp-assignment-numpy, project started by Thomas Moreau and Alexandre Gramfort.

## What we want you to learn by doing this assignment:

  - Use Git and GitHub
  - Work with Python files (and not just notebooks!)
  - Do a pull request on a GitHub repository
  - Format your code properly using standard Python conventions
  - Make your code pass tests run automatically on a continuous integration system (GitHub actions)

## How?
### 1st part: basic PR
  - Fork the repository by clicking on the `Fork` button on the upper right corner
  - Clone the repository of your fork with: `git clone https://github.com/MYLOGIN/github-assignment` (replace MYLOGIN with your GitHub login)
  - Check which remote (distant) repositories your local repository knows, with `git remote -v` (v for verbose)
  - Setup git to work with the original repo (mine) with `git remote add upstream https://github.com/mathurinm/github-assignment` (tell git: there's a new remote repository to track and its name is `upstream`)
  - Check again which remote (distant) repositories your local repositories knows. You should now see 2.
  - Create a branch called `MYNAME` using `git switch -c MYNAME`, where your replace `MYNAME` by YOUR name. Git tells you that it has created a new branch and you are now on this branch.
  - Modify the file `students.txt` by adding an X at the end of the line with your name.
  - Add and commit this file using `git add` and `git commit`. Check what you are doing with `git status`.
  - Push your branch to your repository with `git push`. If you haven't done anything more, git will complain that it does not know where to push: follow the instructions that it displays.
  - Send a Pull Request of this branch to this repository (mathurinm/github-assignment; not your fork) by going to https://github.com/mathurinm/github-assignment/pulls
  - If Github says that there is a conflict (it cannot merge the branch because you modified lines that were modified by someone else at the same time), solve them locally by pulling `upstream main` into your branch (see below how to pull `upstream main`)
  - Ping me on Slack so that I merge the PR. This will allow the test suite (Continuous Integration) to be run on the second part of the assignment.
  - **Once your PR has been merged**: Go back to your `main` branch with `git checkout main`.
  - Delete the branch you used to do the PR (you no longer need it) wiht `git branch -D yourbranchname`
  - Synchronize your local main branch with the main branch of the fork with: `git pull upstream main` (from which remote repo to pull, and which branch)
  - you're set!

### 2nd part: advanced PR
  - Make sure to follow the above steps in **Once your PR has been merged**
  - Create a branch called `MYNAME_assignment` using `git checkout -b MYNAME_assignment` (where `MYNAME` is replaced by your real last name)
  - Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.
  - Open the pull request on GitHub. **The name of the PR should contain your first and last names**.
  - Keep pushing to your branch (the same one) until the continuous integration (CI) system is green. **There is no need to open a new pull request every time you push**. Alternatively, you can run the tests locally with the following command:
  ```
    flake8 .
    pydocstyle
  ```
  - When it is green notify the professors that you are done.
