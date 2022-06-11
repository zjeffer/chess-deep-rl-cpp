FROM ghcr.io/zjeffer/devcontainer:latest

WORKDIR /app

# libtorch
RUN pacman -S --noconfirm wget unzip
RUN wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip \
	&& unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip \
	&& rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip \
	&& mv libtorch /usr/include/libtorch

# compilers, opencv, nvidia 
RUN pacman -Syyu --noconfirm cmake make gcc opencv git vtk hdf5 glew

# add user
################# USER & GROUP CREATION #################
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
################# SUDO #################
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME
################# ZSH #################
ENV TERM xterm
ENV SHELL /bin/zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
RUN sudo cp /root/.zshrc /home/$USERNAME/.zshrc

RUN sudo chown -R $USERNAME:$USERNAME /app

# g3log
COPY ./vendor/g3log/ ./vendor/g3log/
RUN sudo mkdir /app/vendor/g3log/build
RUN sudo chown -R user: ./vendor/g3log/*

RUN cd /app/vendor/g3log/build && cmake .. && sudo cmake --build . --target install

RUN paru -S cuda --noconfirm
RUN paru -S cudnn --noconfirm
RUN paru -S qt5 --noconfirm
RUN paru -S openmpi --noconfirm
RUN paru -S fmt --noconfirm

COPY . .


