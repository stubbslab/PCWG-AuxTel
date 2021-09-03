**IMPORTANT - READ FIRST:**  
This guide is for setting up the current nominal working environment.
If you have made code changes to the packages mentioned which you want to keep: **do not run these instructions**.
If you're a DESC person doing analysis work, or are unsure if this applies to you, it most likely doesn't, so don't panic, and read on :)

Rough work-flow:
1. Open Cisco VPN, connect to NCSA
2. Open the RSP: https://lsst-lsp-stable.ncsa.illinois.edu/
3. Select the correct container version (see below), and at least a Medium container
4. Open a terminal and setup the stack (see below)
5. 
    * First time: follow the setup instructions for cloning and setting up packages
    * subsequent times: you only need to do the `git checkout <branchname>` (if the branch has changed), but then _always_ `git fetch --all` and `git reset --hard origin/tickets/DM-<nnnnn>`.
6. Check your `.user_setups` file (see below for instructions)

These are the package versions currently required for interfacing with the data.
Merlin endeavors to keep this up-to-date as ticket versions change and branches merge, but there will surely be times when I forget.
So, if things aren't working:

1. Check that you've got all these packages checked out, _and_ setup so that they're being used by your kernel (_i.e._ check your notebooks.user_setups file is correct).
2. Check that you're on the right branches of all the packages listed below.
3. Check that you've done the `git fetch --all` and `git reset --hard` parts, as these are the parts that will pull in the changes.
    - if you do a git pull and it doesn't Just Work™ *AND* you haven't made any code changes in that package, then you can almost certainly fix the problem by doing:

Stack setup
-----------
In a terminal, just do:
```
source ${LOADSTACK}
setup lsst_distrib
```

User setups file
----------------
Edited, using your favorite command line editor, with  
`<vi/emacs/ed> ${HOME}/notebooks/.user_setups`  
You should have one line per package we're setting up for each package listed below (except Spectractor), so your file should look like:
```
setup -j daf_butler -r $HOME/repos/daf_butler
setup -j atmospec -r $HOME/repos/atmospec
setup -j rapid_analysis -r $HOME/repos/rapid_analysis
```
NB: You do not need to include (and must _not_ include) an entry like that for Spectractor [^1] , or any other packages, unless you know what you're doing and it's intentional.

Versions: packages, the stack, reductions
-----------------------------------------
List of packages and their associated tickets:  
```
daf_butler: tickets/DM-31623_pin
atmospec: master
rapid_analysis: tickets/DM-31522
Spectractor: tickets/DM-29598
```
Currently recommended stack version: `w_2021_36`  
Currently recommended rerun location for processed data: `/project/shared/auxTel/rerun/mfl/slurmRun/`


Pseudoscript
------------

The following pseudo-script of the packages listed above is given for convenience in a such a form that you can probably copy and paste it in.
Packages and relevant tickets in pseudo-script form:

```
mkdir -p $HOME/repos

cd $HOME/repos
git clone https://github.com/lsst-dm/Spectractor.git
cd Spectractor
git fetch --all
git reset --hard origin/tickets/DM-29598
git pull
pip install -r requirements.txt
pip install -e .

cd $HOME/repos
git clone https://github.com/lsst/daf_butler.git
cd daf_butler
git fetch --all
git reset --hard origin/tickets/DM-31623_pin
setup -j -r .
scons opt=3 -j 4

cd $HOME/repos
git clone https://github.com/lsst-dm/atmospec.git
cd atmospec
git fetch --all
git reset --hard origin/master
setup -j -r .
scons opt=3 -j 4

cd $HOME/repos
git clone https://github.com/lsst-sitcom/rapid_analysis.git
cd rapid_analysis
setup -j -r .
scons opt=3 -j 4
git fetch --all
git reset --hard origin/tickets/DM-31522
```

Footnotes
---------
[^1]: Spectractor and the `.user_setups` file: the entries in that file tell your notebook kernel which package versions to setup for packages managed via `eups`. Spectractor is currently installed via `pip` and therefore does not need an entry. This will likely change in the future, at which point these instructions will be updated accordingly.

Because this is a work-in-progress and you're building ticket branches, the `scons`, which runs the tests, won't necessarily always return without errors, and that sometimes I put it before the git checkout, and sometimes after, depending on whether I think it will succeed or not.
