#! /bin/bash

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    wget http://scip.zib.de/download/release/SCIPOptSuite-$VERSION-Linux.deb
    sudo apt-get update && sudo apt install -y ./SCIPOptSuite-$VERSION-Linux.deb
else
    wget http://scip.zib.de/download/release/SCIPOptSuite-$VERSION-Darwin.dmg
    sudo hdiutil attach ./SCIPOptSuite-$VERSION-Darwin.dmg
fi
