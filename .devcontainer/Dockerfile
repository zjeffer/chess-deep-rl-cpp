FROM ghcr.io/zjeffer/docker-arch-cuda:latest

WORKDIR /app

# compilers, opencv, nvidia 
RUN pacman -Syu --noconfirm cmake make gcc gdb opencv git vtk hdf5 glew openmpi fmt sudo qt5-base qt6 ninja

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
RUN sudo pacman -Sy --noconfirm zsh
ENV TERM xterm
ENV SHELL /bin/zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
ADD .zshrc $HOME

# g3log
COPY ./vendor/g3log/ ./vendor/g3log/
RUN sudo chown -R $USERNAME:$USERNAME /app
RUN mkdir /app/vendor/g3log/build
RUN cd /app/vendor/g3log/build && cmake .. && sudo cmake --build . --target install

COPY . .

