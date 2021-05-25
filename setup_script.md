These are the package versions currently required for interfacing with the data.
I (Merlin) endeavor to keep this up-to-date as ticket versions change and branches merge, but there will surely be times when I forget.
So, if things aren't working:

1) Check that you've got all these packages checked out, _and_ setup so that they're being used by your kernel (_i.e._ in your notebooks.user_setups file)
2) Check that you're on the right branches of all the packages listed below
3) Check that you've done a git pull
    - if you do a git pull and it doesn't Just Work™ *AND* you haven't made any code changes in that package, then you can almost certainly fix the problem by doing:
    git reset --hard origin && git checkout tickets/\<DM-whatever\> && git pull
    but do note that if you've made code changes to the repo that these will be lost.

The following list of packages and ticket branches is given in a such a form that you can copy and paste them in.
Do note that because this is a work-in-progress and you're building off of ticket branches that the scons line won't always return without errors, and that sometimes I put it before the git checkout, and sometimes after, depending on whether I think it will succeed or not.

Currently recommended stack version: w_2021_21
Currently recommended rerun location for processed data: /project/shared/auxTel/rerun/mfl/binning4/  

Packages and relevant tickets in pseudo-script form:


mkdir -p $HOME/repos

cd $HOME/repos  
git clone https://github.com/lsst-dm/Spectractor.git  
cd Spectractor  
git fetch --all  
git reset origin/tickets/DM-29598 --hard  
git pull  
pip install -r requirements.txt  
pip install -e .  


cd $HOME/repos  
git clone https://github.com/lsst/obs_base.git  
cd obs_base  
setup -j -r .  
git fetch --all  
git reset origin/tickets/DM-26719 --hard  
scons opt=3 -j 4  


cd $HOME/repos  
git clone https://github.com/lsst/obs_lsst.git  
cd obs_lsst  
setup -j -r .  
git fetch --all  
git reset origin/tickets/DM-26719 --hard  
scons opt=3 -j 4  


cd $HOME/repos  
git clone https://github.com/lsst-dm/atmospec.git  
cd atmospec  
git fetch --all  
git reset origin/tickets/DM-26719 --hard  
setup -j -r .  
scons opt=3 -j 4  


cd $HOME/repos  
git clone https://github.com/lsst-sitcom/rapid_analysis.git  
cd rapid_analysis  
setup -j -r .  
scons opt=3 -j 4  
git fetch --all  
git reset origin/tickets/DM-21412 --hard  

