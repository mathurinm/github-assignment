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

## How to contribute to an existing repository
### Preliminary steps
If it is your first time using Git and GitHub, you will need to create a [GitHub](https://github.com/) account and [download, install and configure Git](https://docs.github.com/en/get-started/quickstart/set-up-git#setting-up-git). Once this is done, [authenticate with GitHub](https://docs.github.com/en/get-started/quickstart/set-up-git#authenticating-with-github-from-git) with either HTTPS or SSH. The following assumes that you are [cloning with HTTPS URLs](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls). If you are using [SSH URLs](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-ssh-urls), change `https://github.com/` with `git@github.com:` in what follows.

### Contribution
  - Fork the repository at https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME
  - Clone your forked repository: `git clone https://github.com/MYLOGIN/REPOSITORY_NAME`
  - Change directory to `REPOSITORY_NAME`: `cd REPOSITORY_NAME`
  - Add reference to the upstream repository: `git remote add upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME`
  - Check your remote repositories: `git remote -v`. You should see something like this:
    ```
    origin https://github.com/MYLOGIN/REPOSITORY_NAME.git (fetch)
    origin https://github.com/MYLOGIN/REPOSITORY_NAME.git (push)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (fetch)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (push)
    ```
  - Fetch the `main` branch from the upstream repository: `git fetch upstream main`
  - Go on the upstream `main` branch: `git checkout upstream/main`
  - Make sure you are up-to-date with the upstream `main` branch: `git pull upstream main`
  - Create and switch to a new working branch: `git switch -c MYBRANCH`
  - Make the desired modifications
  - Add the modified file(s) to commit: `git add FILE` (use space to add multiple files)
  - Create a commit with a message: `git commit -m "RELEVANT MESSAGE"`
  - Push your commit on the upstream repository from your working branch: `git push --set-upstream origin MYBRANCH`. Now, to create a pull request, you can either click the link on your terminal or go to the web page of your repository and click the `Compare & pull request` button
  - Click the `Create pull request` button
  - You're set!
