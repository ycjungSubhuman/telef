FROM heichblatt/archlinux-yaourt

RUN /bin/bash -c "yaourt -Syy"
RUN /bin/bash -c "yaourt -S --noconfirm clang nvidia cuda" 
COPY ./flann /home/user/flann
RUN /bin/bash -c "sudo chown user:user /home/user/flann"
RUN /bin/bash -c "cd /home/user/flann && makepkg -si --noconfirm"
RUN /bin/bash -c "rm -rf /home/user/flann"

RUN /bin/bash -c "yaourt -S --noconfirm eigen"
RUN /bin/bash -c "yaourt -S --noconfirm vtk glew gtest openni2-git netcdf proj gdal lz4 openmpi tk unixodbc hdf5 mariadb postgresql-libs"

COPY ./pcl-git /home/user/pcl-git
RUN /bin/bash -c "sudo chown user:user /home/user/pcl-git"
RUN /bin/bash -c "cd /home/user/pcl-git && makepkg -si --noconfirm"
RUN /bin/bash -c "rm -rf /home/user/pcl-git"

COPY ./ceres-solver /home/user/ceres-solver
RUN /bin/bash -c "sudo chown user:user /home/user/ceres-solver"
RUN /bin/bash -c "cd /home/user/ceres-solver && makepkg -si --noconfirm"
RUN /bin/bash -c "rm -rf /home/user/ceres-solver"

RUN /bin/bash -c "yaourt -S --noconfirm opencv2"

RUN /bin/bash -c "yaourt -Scc"

COPY ./CLion-2018.1.5.tar.gz /home/user/clion.tar.gz
RUN sudo chown user:user /home/user/clion.tar.gz && \
    cd /home/user/ && tar -xvf clion.tar.gz && \
    sudo ln -s /home/user/clion-2018.1.5/bin/clion.sh /usr/bin/clion

COPY ./OpenNI2-FreenectDriver /home/user/OpenNI2-FreenectDriver
RUN /bin/bash -c "sudo cp /home/user/OpenNI2-FreenectDriver/* /usr/lib/OpenNI2/Drivers/ && sudo rm -rf /home/user/OpenNI2-FreenectDriver"
RUN /bin/bash -c "yaourt -S --noconfirm openni"

COPY ./ncurses5-compat-libs /home/user/ncurses5-compat-libs
RUN /bin/bash -c "sudo chown user:user /home/user/ncurses5-compat-libs"
RUN /bin/bash -c "cd /home/user/ncurses5-compat-libs && makepkg -si --noconfirm"
RUN /bin/bash -c "yaourt -S --noconfirm gdb"
RUN /bin/bash -c "rm -rf /home/user/ncurses5-compat-libs"

RUN /bin/bash -c "cd /home/user && git clone https://github.com/FloopCZ/tensorflow_cc"
RUN /bin/bash -c "yaourt -S --noconfirm unzip mlocate python python-numpy wget gcc7 bazel cudnn"
RUN /bin/bash -c "yaourt -S --noconfirm dlib"
RUN /bin/bash -c "cd /home/user/tensorflow_cc/tensorflow_cc && mkdir build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ../ && make && sudo make install && rm -rf /home/user/tensorflow_cc"
