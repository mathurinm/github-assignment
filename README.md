# Working with git and GitHub
This README contains:
- an assignment for students of the Python class
- general useful information about working with GitHub.

## I) The assignment
This work is based on https://github.com/x-datascience-datacamp/datacamp-assignment-numpy, project started by Thomas Moreau and Alexandre Gramfort.

### What this assignment teaches you:

  - Use git and GitHub
  - Work with Python files (and not just notebooks!)
  - Do a pull request on a GitHub repository
  - Format your code properly using standard Python conventions
  - Make your code pass tests run automatically on a continuous integration system (GitHub actions)



### Prerequisite
  - Create a GitHub account.
  - Add **one** of the following ways to authenticate to push to your repositories:
    - HTTPS: generate and save a GitHub token to connect with HTTPS: [GitHub tutorial](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
    - SSH: generate and add a public/private SSH key pair to GitHub. [GitHub tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

    (SSH may be a bit more complex)
### Part A: basic PR
  - Fork this repository by clicking on the `Fork` button on the upper right corner of this page.
  - Clone the repository of your fork with: `git clone https://github.com/YOURLOGIN/github-assignment` (replace YOURLOGIN with your GitHub login)
  - Check your directory with `pwd` then access the GitHub repository folder with `cd github-assignment`
  - Check which remote (distant) repositories your local repository knows, with `git remote -v` (v for verbose)
  - Setup git to work with the original repo (mine) with `git remote add upstream https://github.com/mathurinm/github-assignment` (tell git: there's a new remote repository to track and its name is `upstream`)
  - Check again which remote (distant) repositories your local repositories knows (with `git remote -v`). You should now see 2.
  - Create a branch called `YOURNAME` using `git switch -c YOURNAME`, where your replace `YOURNAME` by YOUR name. Git tells you that it has created a new branch and you are now on this branch.
  - Check the GitHub repository where you are and on which branch with `git status`.
  - Modify the file `students.txt` by adding an `X` at the end of the line with your name. This should be done on your machine, using vscode for example.
  - Add and commit this file using `git add THEFILEYOUWANTTOADD` and `git commit m "YOUR COMMIT MESSAGE HERE"`. Check what you are doing with `git status`.
  - Push your branch to your repository with `git push`. If you haven't done anything more, git will complain that it does not know where to push: follow the instructions that it displays. If it complains that you must login, check the Prerequisite section of this Readme.
  - Send a Pull Request of this branch to this repository (mathurinm/github-assignment; not your fork) by going to https://github.com/mathurinm/github-assignment/pulls
  - If GitHub says that there is a conflict (it cannot merge the branch because you modified lines that were modified by someone else at the same time), solve them locally by pulling `upstream main` into your branch (see below how to pull `upstream main`)
  - Ping me on Slack so that I merge the PR. This will allow the test suite (Continuous Integration) to be run on the second part of the assignment. Otherwise, CI is not run on PRs from people who have never contributed to the repo before.
  - **Once your PR has been merged**: Switch back to your `main` branch with `git switch main`.
  - Delete the branch you used to do the PR (you no longer need it) with `git branch -D YOURBRANCHNAME`
  - Synchronize your local main branch with the main branch of the fork with: `git pull upstream main` (from which remote repo to pull, and which branch)
  - You're set for the second part!

### Part B: advanced PR
  - Make sure to follow the above steps in **Once your PR has been merged** above.
  - Create a branch called `YOURNAME_assignment` using `git switch -c YOURNAME_assignment` (where `YOURNAME` is replaced by your real last name)
  - Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.
  - Open the pull request on GitHub. **The name of the PR should contain your first and last names**.
  - Keep pushing to your branch (the same one) until the continuous integration (CI) system is green. **There is no need to open a new pull request every time you push**. Be aware that you can also run the tests locally with the following command:
  ```
    flake8 .
    pydocstyle
  ```
  - When the CI is green notify the professors that you are done.


# General information

## Summary: how to contribute to an existing repository
  - Fork the repository you want to contribute to at https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME
  - Clone your forked repository: `git clone https://github.com/YOURLOGIN/REPOSITORY_NAME` or `git clone git@github.com:YOURLOGIN/REPOSITORY_NAME` (depending on if you want to use the HTTPS or the SSH protocol)
  - Change directory to `REPOSITORY_NAME`: `cd REPOSITORY_NAME`
  - Add reference to the repository named upstream: `git remote add upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME` (or `git@github.com:REPOSITORY_OWNER/REPOSITORY_NAME`)
  - (optional sanity check) Check your remote repositories: `git remote -v`. You should see something like this:
    ```
    origin https://github.com/YOURLOGIN/REPOSITORY_NAME.git (fetch)
    origin https://github.com/YOURLOGIN/REPOSITORY_NAME.git (push)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (fetch)
    upstream https://github.com/REPOSITORY_OWNER/REPOSITORY_NAME (push)
    ```
    (or the same with `git@github.com:...` if you added them with the SSH syntax)
  - Locally, switch to the upstream `main` branch: `git switch main`
  - Make sure you are up-to-date with the upstream `main` branch: `git pull upstream main`
  - Create and switch to a new working branch called `YOURBRANCH` (name to customize): `git switch -c YOURBRANCH` (`-c` means "create").
  - Make the desired modifications to your files
  - Add the modified file(s) to commit: `git add FILE` (use space to add multiple files: `git add FILE1 FILE2`)
  - Create a commit with a message: `git commit -m "RELEVANT COMMIT NAME"`. The commit name should be a meaningful summary of the modifications that the commit contains (what was added/removed/changed/fixed/...).
  - Push your commit on the upstream repository from your working branch: `git push --set-upstream origin YOURBRANCH`. Note that you only need to use `--set-upstream origin YOURBRANCH` once: after that, git remembers that your local branch should be pushed to the corresponding distant branch.  Now, to create a pull request, you can either click the link that appears on your terminal or go to the web page of your repository and click the `Compare & pull request` button
  - Click the `Create pull request` button, select which branch you want to merge into which branch
  - You're set! You can continue working locally on the same branch, and push your changes regularly. the Pull Request page on GitHub will update every time you push.


## Good practices with git
- When working on a big feature, avoid pushing to your `main` directly. Instead, sending a PR allows other contributors to review your work easily. **This holds even when you are working alone on a single repository**: you can send PRs from one branch of the repo to the `main` branch. Keeping stuff separated allows you to work on separate features at the same time, while preserving a clean and working `main`.
- Use meaningful names for commits, branches, and PR titles: they are a way to communicate with your collaborators, or your future self. Commit names should not be longer than 80 characters.
- Commit often, even when it's a work in progress (WIP). This way, other contributors can see your work and give early feedback. It's not an issue to push unfinished work to a PR, that's wat they are for.
- One PR should address one and only one issue. You should avoid "hanging PRs" that tackle problems which are too big and end up never being merged (thus work was done in vain). Split big tasks into smaller ones; this also makes reviewing of PRs easier.
- If somebody sends a PR to your repo and you want to make changes to it, you can get it locally with the `gh` tool (installable with `conda install gh`): `gh pr checkout <PR_NUMBER>`. There also exist built in ways to do this with `git` but the syntax is more complex to remember.
- Use squash commits when you merge PRs. This squashes all intermediate, sometimes meaningless or incomplete, commits of the PR into a single commit when merging into main. Thus the history of `main` is composed of complete commits. Delete the list of commits from the commit message, but keep the "coauthored by" section at the end for credit attribution to eventual other contributors of the PR.


## git commands you should understand

git is a powerful tool with a nonzero entry cost, that has many complex possibilities. However, you can (should?) leverage a good fraction of its power knowing only a handful of commands:


- `git clone ADDRESS` where `ADDRESS` is copied from github when you hit the "clone" button.
When you clone, the repo is downloaded in your current working directory. You must then use `cd` to enter it.
- `git config --global user.name "YOURNAME"`: globally sets your username (to do once only, when you start using git on a new machine); `--global` is an option, telling git to set this configuration for every git project on your computer.
You can put any name as user name, it does have to match your GitHub name. It's the name which will appear in the commit list.
You should also configure your email with
`git config --global user.email "YOUREMAILADDRESS"`.
- `git log -4`: show the last 4 commits on your local branch. Use enter to navigate the editor that this commands opens (that is called [`less`](https://stackoverflow.com/questions/9483757/how-to-exit-git-log-or-git-diff), and press `q` if you want to exit it.
- `git add FILENAME` start tracking a new file, or stage a currently tracked (and modified since last commit) file to be committed in the next commit.
- `git commit -m "YOUR COMMIT MSG"` creates a commit including all currently staged files.
- `git push`: publish your commit on the remote repository, on the corresponding branch.
- `git pull`: get the latest commits from the corresponding branch on the remote repository.
- `git remote -v`: see the address and alias of the associated remote repository (potentially, repositories).
- `git fetch`, `git rebase`: similar ideas to pull and merge, but done in a different fashion. See [this StackOverflow question](url{https://stackoverflow.com/questions/16666089/whats-the-difference-between-git-merge-and-git-rebase). be aware of those, but usually you'll be fine/better with simple merging.
- `git reset --hard HEAD`: undoes the changes of the latest commit. Do not do it if you have already pushed your last commit, as you won't be able to push in a clean fashion. If you want to undo an already published commit, use
`git revert YOUR_COMMIT_HASH`, where `YOUR_COMMIT_HASH` is a number like 444b1cff (get it via the `git log` command mentioned above)



# VSCode configuration
- Install the VScode `Python` and `autopep8` extensions vie the Extensions menu (`ctrl + shift + x` or left panel)
- Configure your editor to autoformat on save, and to use autopep8 as formatter, by adding the following options to your User settings (Type `ctrl + maj + p` to open the command palette, then search for "Preferences: open User settings (JSON)" to open your `settings.json`):

```
    "files.trimTrailingWhitespace": true,
    "editor.formatOnSave": true,
    "[python]": {
        "editor.rulers": [
            79
        ],
        "editor.semanticHighlighting.enabled": false,
        "editor.formatOnType": true,
        "editor.defaultFormatter": "ms-python.autopep8",
    },
    "autopep8.args": [
        "--max-line-length=88"
    ],
```
