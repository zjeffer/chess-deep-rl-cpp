# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="agnoster"


# plugins
plugins=(git)

source $ZSH/oh-my-zsh.sh

# aliases
alias ls='ls --color=auto'
alias la='ls -lah'
alias ll='ls -l'
alias free='free -h'
alias df='df -h'
alias ka='killall'
alias gs='git status'
alias less='less -I '
