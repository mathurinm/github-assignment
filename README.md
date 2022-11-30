# Working with git and GitHub
This README contains:
- an assignment for students of the Python class
- general useful information about working with GitHub.

## I) The assignment
Forked from https://github.com/x-datascience-datacamp/datacamp-assignment-numpy, project started by Thomas Moreau and Alexandre Gramfort.

### What this assignment teaches you:

  - Use git and GitHub
  - Work with Python files (and not just notebooks!)
  - Do a pull request on a GitHub repository
  - Format your code properly using standard Python conventions
  - Make your code pass tests run automatically on a continuous integration system (GitHub actions)



### Prerequisite
  - Create a GitHub account.
### Part A: basic PR
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
  - Ping me on Slack so that I merge the PR. This will allow the test suite (Continuous Integration) to be run on the second part of the assignment. Otherwise, CI is not run on PRs from people who have never contributed to the repo before.
  - **Once your PR has been merged**: Go back to your `main` branch with `git checkout main`.
  - Delete the branch you used to do the PR (you no longer need it) with `git branch -D yourbranchname`
  - Synchronize your local main branch with the main branch of the fork with: `git pull upstream main` (from which remote repo to pull, and which branch)
  - You're set!

### Part B: advanced PR
  - Make sure to follow the above steps in **Once your PR has been merged** above.
  - Create a branch called `MYNAME_assignment` using `git switch -c MYNAME_assignment` (where `MYNAME` is replaced by your real last name)
  - Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.
  - Open the pull request on GitHub. **The name of the PR should contain your first and last names**.
  - Keep pushing to your branch (the same one) until the continuous integration (CI) system is green. **There is no need to open a new pull request every time you push**. Alternatively, you can run the tests locally with the following command:
  ```
    flake8 .
    pydocstyle
  ```
  - When the CI is green notify the professors that you are done.


# General information

## Summary: how to contribute to an existing repository
  - Fork the repository you want to contribute to at https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME
  - Clone your forked repository: `git clone https://github.com/MYLOGIN/REPOSITORY_NAME` or `git clone git@github.com:MYLOGIN/REPOSITORY_NAME` (depending on if you use the HTTPS or the SSH protocol)
  - Change directory to `REPOSITORY_NAME`: `cd REPOSITORY_NAME`
  - Add reference to the repository named upstream: `git remote add upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME`
  - (optional sanity check) Check your remote repositories: `git remote -v`. You should see something like this:
    ```
    origin https://github.com/MYLOGIN/REPOSITORY_NAME.git (fetch)
    origin https://github.com/MYLOGIN/REPOSITORY_NAME.git (push)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (fetch)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (push)
    ```
    (or the same with `git@github.com:...` if you added them with the SSH syntax)
  - Fetch the `main` branch from the upstream repository: `git fetch upstream main`
  - Locally, switch to the upstream `main` branch: `git checkout upstream/main`
  - Make sure you are up-to-date with the upstream `main` branch: `git pull upstream main`
  - Create and switch to a new working branch called `MYBRANCH` (name to customize): `git switch -c MYBRANCH`
  - Make the desired modifications to your files
  - Add the modified file(s) to commit: `git add FILE` (use space to add multiple files)
  - Create a commit with a message: `git commit -m "RELEVANT MESSAGE"`
  - Push your commit on the upstream repository from your working branch: `git push --set-upstream origin MYBRANCH`. Now, to create a pull request, you can either click the link that appears on your terminal or go to the web page of your repository and click the `Compare & pull request` button
  - Click the `Create pull request` button, select which branch you want to merge into which branch
  - You're set!


## Good practices with git
- When working on a big feature, avoid pushing to `main` directly. Instead, sending a PR allows other contributors to review your work easily. **This holds even when you are working alone on a single repository**: you can send PRs from one branch of the repo to the `main` branch. Keeping stuff separated allows you to work on separate features at the same time, while repserving a clean and working `main`.
- Use meaningful names for commits, branches, and PR titles: they are a way to communicate with your collaborators, or your future self. Commit names should not be longer than 80 characters.
- One PR should address one and only one issue. You should avoid "hanging PR" that tackle problems which are too big and end up never being merged (thus work was done in vain). Split big tasks into smaller ones; this also makes reviewing of PRs easier.
- If somebody sends a PR to your repo and you want to make changes to it, you can get it locally with the `gh` tool (installable with `conda install gh`): `gh pr checkout <PR_NUMBER>`. There also exist built in ways to do this with `git` but the syntax is more complex to remember.
- Use squash commits when you merge PRs. This squashes all intermediate, sometimes meaningless or incomplete, commits of the PR into a single commit when merging into main. Thus the history of `main` is composed of complete commits. Delete the list of commits from the commit message, but keep the "coauthored by" section at the end for credit attribution to eventual other contributors of the PR.
- Commit often, even when it's a work in progress (WIP). This way, other contributors can see your work and give early feedback. It's not an issue to push unfinished work to a PR, that's wat they are for.
