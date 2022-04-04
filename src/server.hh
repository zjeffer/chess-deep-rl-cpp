#ifndef server_hh
#define server_hh

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

class Server{
   public:
    Server(uint16_t port);

    void run();


};

#endif /* server_hh */