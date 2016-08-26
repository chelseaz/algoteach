# Installation: Second try

For a more painless installation, I gave in and created a Ubuntu 14.04 desktop VM with Vagrant:

    vagrant init box-cutter/ubuntu1404-desktop

For Ubuntu 12.04, replace 1404 with 1204.

I allocated 4GB of memory to the VM. (With 1GB, building OpenRAVE fails.)

To keep guest additions in sync, install this [plugin](http://kvz.io/blog/2013/01/16/vagrant-tip-keep-virtualbox-guest-additions-in-sync/).

I also had to [disable 3D acceleration](https://github.com/vDevices/Vagrant-LinuxDesktop/blob/master/Vagrantfile) to avoid [OpenGL warnings](http://stackoverflow.com/questions/20031535/opengl-warning-from-a-net-program) and broken GUIs. 

## ROS

I just followed the instructions at ROS.org. Indigo on Ubuntu 14.04 went smoothly.

For Hydro on Ubuntu 12.04, I had to deal with [this](http://askubuntu.com/questions/41605/trouble-downloading-packages-list-due-to-a-hash-sum-mismatch-error) and [this](https://ubuntuforums.org/archive/index.php/t-2205536.html)

## OpenRAVE

Official Linux installation instructions are at http://openrave.org/docs/latest_stable/coreapihtml/installation_linux.html

I had to re-install ca-certificates: http://www.webupd8.org/2014/03/fix-cannot-add-ppa-please-check-that.html

The `sudo apt-get build-dep openrave` step failed, so I instead followed [these instructions](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html) for installing OpenRAVE from source. 

If on Ubuntu 14.04, you can build a more recent version of OpenRAVE than the commit that link referenced. The instructions don't include installing [fcl](https://github.com/flexible-collision-library/fcl/tree/fcl-0.5). Check out the `0.5.0` tag before building from source.

## TrajOpt

I followed http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/install.html, skipping Gurobi for now. As in http://lfd.readthedocs.io/en/latest/install.html, I checked out the trajopt-jointopt branch of [this fork](https://github.com/erictzeng/trajopt) and ran `cmake` like so:

    cmake /path/to/trajopt -DBUILD_CLOUDPROC=ON

## BulletSim

First install libhdf5-dev, then follow the instructions at https://github.com/hojonathanho/bulletsim

