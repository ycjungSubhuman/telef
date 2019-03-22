FROM heichblatt/archlinux-yaourt

RUN yaourt -Syy
RUN yaourt -S --noconfirm clang nvidia cuda eigen vtk glew gtest openni2-git netcdf proj gdal lz4 openmpi tk unixodbc hdf5 mariadb postgresql-libs opencv2 openni gdb unzip mlocate python python-numpy wget gcc7 bazel cudnn dlib glfw-x11
COPY ./flann /home/user/flann
RUN sudo chown user:user /home/user/flann && \
    cd /home/user/flann && \
    makepkg -si --noconfirm && \
    rm -rf /home/user/flann

COPY ./pcl-git /home/user/pcl-git
RUN sudo chown user:user /home/user/pcl-git && \
    cd /home/user/pcl-git && \
    makepkg -si --noconfirm && \
    rm -rf /home/user/pcl-git

COPY ./ceres-solver /home/user/ceres-solver
RUN sudo chown user:user /home/user/ceres-solver && \
    cd /home/user/ceres-solver && \
    makepkg -si --noconfirm && \
    rm -rf /home/user/ceres-solver

COPY ./CLion-2018.1.5.tar.gz /home/user/clion.tar.gz
RUN sudo chown user:user /home/user/clion.tar.gz && \
    cd /home/user/ && tar -xvf clion.tar.gz && \
    sudo ln -s /home/user/clion-2018.1.5/bin/clion.sh /usr/bin/clion

COPY ./OpenNI2-FreenectDriver /home/user/OpenNI2-FreenectDriver
RUN sudo cp /home/user/OpenNI2-FreenectDriver/* /usr/lib/OpenNI2/Drivers/ && \
    sudo rm -rf /home/user/OpenNI2-FreenectDriver

COPY ./ncurses5-compat-libs /home/user/ncurses5-compat-libs
RUN sudo chown user:user /home/user/ncurses5-compat-libs && \
    cd /home/user/ncurses5-compat-libs && \
    makepkg -si --noconfirm && \
    rm -rf /home/user/ncurses5-compat-libs

RUN cd /home/user && \
    git clone https://github.com/FloopCZ/tensorflow_cc && \
    cd /home/user/tensorflow_cc/tensorflow_cc && \
    mkdir build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ../ && \
    make && sudo make install && \
    rm -rf /home/user/tensorflow_cc

RUN yaourt -Scc
