My notes for installing robotics software on OSX Yosemite.

## Installing ROS

To install on OSX, I followed
https://github.com/mikepurvis/ros-install-osx

I wasn't sure what to do with the `indigo_desktop_full_ws` directory that it created, so I deleted it.

I needed to work in the bash shell, especially when doing `source devel/setup.bash` in the workspace, or
`source /opt/ros/indigo/setup.bash`.


## Installing Gazebo

I used the [one-line script](http://gazebosim.org/tutorials?tut=install_on_mac&cat=install), since `brew install gazebo7` resulted in `DYLD_LIBRARY_PATH` problems.


## Installing OpenRAVE

First I installed [MacPorts](https://www.macports.org/install.php). 

Then I followed the [OSX instructions](http://openrave.org/docs/latest_stable/coreapihtml/installation_macosx.html). This required that I first add

    /opt/local/bin
    /opt/local/sbin

to `/etc/paths`, to avoid [errors](http://stackoverflow.com/questions/9694395/sudo-port-command-not-found).

Do `brew install qt coin ffmpeg ode libccd` to install library dependencies.

I also had to install [FCL](https://github.com/flexible-collision-library/fcl). I checked out the fcl-0.5 branch.

Afterward follow the directions for "Building OpenRAVE" [here](http://openrave.org/docs/latest_stable/coreapihtml/installation_linux.html).

To circumvent [this error](https://github.com/rdiankov/openrave/issues/327), I did `brew install crlibm`.

To circumvent [this error](http://stackoverflow.com/questions/11370684/what-is-libintl-h-and-where-can-i-get-it), I tried doing `brew reinstall --universal --with-examples gettext` and `brew link gettext --force`, to no avail. Thinking a MacPorts installation of gettext might be interfering, I uninstalled MacPorts. That didn't work. Instead, I had to modify my `make` command along [these lines](https://wincent.com/wiki/Setting_up_the_Git_documentation_build_chain_on_Mac_OS_X_Leopard):

    make LDFLAGS="-lintl"

I got some errors about undeclared identifiers in ffmpeg, [indicating](https://github.com/dirkvdb/ffmpegthumbnailer/issues/128) OpenRave depends on older versions of ffmpeg (before 3.0). I did:

    brew uninstall ffmpeg
    brew tap homebrew/versions
    brew install ffmpeg28 

At 100% installation, I ran into [this error](http://stackoverflow.com/questions/23284473/fatal-error-eigen-dense-no-such-file-or-directory) and used the symlink solution.

Finally, `make install` [won't work on OSX](https://github.com/antirez/redis/issues/495) unless you rename the `INSTALL` file, e.g. to `INSTALL-temp`. 

At this point I tried to open `openrave`, only to get stuck with the following:

    2016-08-10 16:43:23,075 openrave [WARN] [plugindatabase.h:929 RaveDatabase::_SysLoadLibrary] /usr/local/lib/openrave0.9-plugins/libqtcoinrave.dylib: dlopen(/usr/local/lib/openrave0.9-plugins/libqtcoinrave.dylib, 1): Library not loaded: /usr/local/opt/coin/Frameworks/Inventor.framework/Versions/C/Libraries/Inventor
      Referenced from: /usr/local/lib/openrave0.9-plugins/libqtcoinrave.dylib
      Reason: image not found

Someone with the same problem: http://openrave-users-list.185357.n3.nabble.com/Error-while-running-OpenRAVE-on-OS-X-El-Capitan-td4027753.html

A possible solution, but the `install_name_tool` command didn't work for me: http://forum.freecadweb.org/viewtopic.php?t=6263

Relatedly or unrelatedly, doing `import openravepy` in python results in a segfault.

So ends my attempt to get OpenRAVE working on OSX.

An alternative to working in an Ubuntu VM is Docker, but it's headless and I get errors like

    [qtcoinrave.cpp:44 CreateInterfaceValidated] no display detected, so cannot load viewer[plugindatabase.h:577 Create] Failed to create name qtcoin, interface viewer
    [qtosgravemain.cpp:37 CreateInterfaceValidated] no display detected, so cannot load viewer[plugindatabase.h:577 Create] Failed to create name qtosg, interface viewer
    [qtcoinrave.cpp:44 CreateInterfaceValidated] no display detected, so cannot load viewer[plugindatabase.h:577 Create] Failed to create name QtCoin, interface viewer
    [qtcoinrave.cpp:44 CreateInterfaceValidated] no display detected, so cannot load viewer[plugindatabase.h:577 Create] Failed to create name QtCameraViewer, interface viewer
    [openrave.cpp:315 MainOpenRAVEThread] failed to find an OpenRAVE viewer.


## Links

ROS tutorials:
http://wiki.ros.org/ROS/Tutorials
 
PR2 tutorials:
http://wiki.ros.org/pr2_robot/Tutorials

Gazebo tutorials:
http://gazebosim.org/tutorials