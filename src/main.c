#include <stdio.h>
#include <string.h>
#include "dispatch.h"

void serverMain(int argc, char** argv);
void validation(void);
void memInfo(void);
void ggufInfo(const char* path);

int main(int argc, char** argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "val") == 0) {
        validation();
    } else if (argc > 1 && strcmp(argv[1], "meminfo") == 0) {
        memInfo();
    } else if (argc > 1 && strcmp(argv[1], "ggufinfo") == 0) {
        ggufInfo(argc > 2 ? argv[2] : ".");
    } else {
        serverMain(argc, argv);
    }
    return 0;
}
