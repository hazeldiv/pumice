#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include "safetensors.h"
#include "data.h"

typedef struct {
    const char* p;
    const char* end;
} jctx;

static void jws(jctx* c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t')) c->p++;
}

static int jch(jctx* c) {
    return c->p < c->end ? *c->p : '\0';
}

static void jskip_string(jctx* c) {
    if (jch(c) != '"') return;
    c->p++;
    while (c->p < c->end) {
        char ch = *c->p;
        if (ch == '\\') {
            c->p += (c->p + 1 < c->end) ? 2 : 1;
            continue;
        }
        c->p++;
        if (ch == '"') break;
    }
}

static int jstr(jctx* c, char* out, int cap) {
    if (jch(c) != '"') return 0;
    c->p++;
    int n = 0;
    while (c->p < c->end) {
        char ch = *c->p;
        if (ch == '\\') {
            if (c->p + 1 < c->end) {
                if (n < cap - 1) out[n++] = c->p[1];
                c->p += 2;
            } else {
                c->p++;
            }
            continue;
        }
        if (ch == '"') {
            c->p++;
            break;
        }
        if (n < cap - 1) out[n++] = ch;
        c->p++;
    }
    out[n] = '\0';
    return 1;
}

static long long jnum(jctx* c) {
    char* endp = NULL;
    long long v = strtoll(c->p, &endp, 10);
    c->p = endp;
    return v;
}

static void jskip_value(jctx* c) {
    jws(c);
    char ch = jch(c);
    if (ch == '"') {
        jskip_string(c);
        return;
    }
    if (ch == '{') {
        c->p++;
        while (1) {
            jws(c);
            if (jch(c) == '}') { c->p++; return; }
            char tmp[256];
            jstr(c, tmp, sizeof(tmp));
            jws(c);
            if (jch(c) == ':') c->p++;
            jskip_value(c);
            jws(c);
            if (jch(c) == ',') { c->p++; continue; }
            if (jch(c) == '}') { c->p++; return; }
            return;
        }
    }
    if (ch == '[') {
        c->p++;
        while (1) {
            jws(c);
            if (jch(c) == ']') { c->p++; return; }
            jskip_value(c);
            jws(c);
            if (jch(c) == ',') { c->p++; continue; }
            if (jch(c) == ']') { c->p++; return; }
            return;
        }
    }
    while (c->p < c->end) {
        char cc = *c->p;
        if (cc == ',' || cc == '}' || cc == ']' || cc == ' ' || cc == '\n' || cc == '\r' || cc == '\t') break;
        c->p++;
    }
}

static sa_dtype dtype_parse(const char* s) {
    if (strcmp(s, "BF16") == 0) return SA_DTYPE_BF16;
    if (strcmp(s, "F32") == 0) return SA_DTYPE_F32;
    if (strcmp(s, "F16") == 0) return SA_DTYPE_F16;
    if (strcmp(s, "I8") == 0) return SA_DTYPE_I8;
    return SA_DTYPE_UNKNOWN;
}

static int parse_shape(jctx* c, sa_tensor* t) {
    jws(c);
    if (jch(c) != '[') return 0;
    c->p++;
    t->ndim = 0;
    while (1) {
        jws(c);
        if (jch(c) == ']') { c->p++; break; }
        int64_t v = jnum(c);
        if (t->ndim < 8) t->shape[t->ndim] = v;
        t->ndim++;
        jws(c);
        if (jch(c) == ',') { c->p++; continue; }
        if (jch(c) == ']') { c->p++; break; }
        return 0;
    }
    return 1;
}

static int parse_offsets(jctx* c, sa_tensor* t) {
    jws(c);
    if (jch(c) != '[') return 0;
    c->p++;
    jws(c);
    t->offset = jnum(c);
    jws(c);
    if (jch(c) == ',') c->p++;
    jws(c);
    long long endOff = jnum(c);
    jws(c);
    if (jch(c) == ']') c->p++;
    t->length = endOff - t->offset;
    return 1;
}

static void append_tensor(safetensors* sf, const sa_tensor* t) {
    if (sf->tensorCount >= sf->tensorCap) {
        sf->tensorCap = sf->tensorCap == 0 ? 2048 : sf->tensorCap * 2;
        sf->tensors = (sa_tensor*)realloc(sf->tensors, sizeof(sa_tensor) * sf->tensorCap);
    }
    sf->tensors[sf->tensorCount++] = *t;
}

static int parse_header(safetensors* sf, const char* buf, size_t len, int fileIndex) {
    jctx c;
    c.p = buf;
    c.end = buf + len;
    jws(&c);
    if (jch(&c) != '{') return -1;
    c.p++;

    while (1) {
        jws(&c);
        if (jch(&c) == '}') { c.p++; break; }
        char name[256];
        if (!jstr(&c, name, sizeof(name))) return -1;
        jws(&c);
        if (jch(&c) != ':') return -1;
        c.p++;
        jws(&c);
        if (strcmp(name, "__metadata__") == 0) {
            jskip_value(&c);
        } else {
            sa_tensor t;
            memset(&t, 0, sizeof(t));
            t.fileIndex = fileIndex;
            snprintf(t.name, sizeof(t.name), "%s", name);
            if (jch(&c) != '{') return -1;
            c.p++;
            while (1) {
                jws(&c);
                if (jch(&c) == '}') { c.p++; break; }
                char key[64];
                if (!jstr(&c, key, sizeof(key))) return -1;
                jws(&c);
                if (jch(&c) != ':') return -1;
                c.p++;
                jws(&c);
                if (strcmp(key, "dtype") == 0) {
                    char dt[16];
                    jstr(&c, dt, sizeof(dt));
                    t.dtype = dtype_parse(dt);
                } else if (strcmp(key, "shape") == 0) {
                    parse_shape(&c, &t);
                } else if (strcmp(key, "data_offsets") == 0) {
                    parse_offsets(&c, &t);
                } else {
                    jskip_value(&c);
                }
                jws(&c);
                if (jch(&c) == ',') { c.p++; continue; }
                if (jch(&c) == '}') { c.p++; break; }
                return -1;
            }
            append_tensor(sf, &t);
        }
        jws(&c);
        if (jch(&c) == ',') { c.p++; continue; }
        if (jch(&c) == '}') { c.p++; break; }
        return -1;
    }
    return 0;
}

int safetensors_open(safetensors* sf, const char** paths, int count) {
    memset(sf, 0, sizeof(*sf));
    if (count > SA_MAX_FILES) return -1;

    for (int i = 0; i < count; i++) {
        sf->files[i] = fopen(paths[i], "rb");
        if (!sf->files[i]) {
            fprintf(stderr, "cannot open weight file: %s\n", paths[i]);
            return -1;
        }
        uint64_t hlen = 0;
        if (fread(&hlen, 8, 1, sf->files[i]) != 1) return -1;
        if (hlen == 0 || hlen > (1ull << 30)) return -1;
        char* hbuf = (char*)malloc((size_t)hlen + 1);
        if (!hbuf) return -1;
        if (fread(hbuf, 1, (size_t)hlen, sf->files[i]) != (size_t)hlen) {
            free(hbuf);
            return -1;
        }
        hbuf[hlen] = '\0';
        int before = sf->tensorCount;
        int rc = parse_header(sf, hbuf, (size_t)hlen, i);
        free(hbuf);
        if (rc != 0) return -1;
        int64_t dataOffset = 8 + (int64_t)hlen;
        for (int k = before; k < sf->tensorCount; k++) {
            sf->tensors[k].offset += dataOffset;
        }
    }
    sf->fileCount = count;
    return 0;
}

const sa_tensor* safetensors_find(const safetensors* sf, const char* name) {
    for (int i = 0; i < sf->tensorCount; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return &sf->tensors[i];
    }
    return NULL;
}

float* safetensors_load_f32(const safetensors* sf, const sa_tensor* t, int64_t* outCount) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    float* out = (float*)malloc(sizeof(float) * n);
    if (!out) return NULL;

    FILE* f = sf->files[t->fileIndex];
    _fseeki64(f, t->offset, SEEK_SET);

    if (t->dtype == SA_DTYPE_F32) {
        fread(out, sizeof(float), (size_t)n, f);
    } else if (t->dtype == SA_DTYPE_BF16) {
        uint16_t chunk[8192];
        int64_t done = 0;
        while (done < n) {
            int64_t c = n - done;
            if (c > 8192) c = 8192;
            size_t got = fread(chunk, sizeof(uint16_t), (size_t)c, f);
            if (got != (size_t)c) {
                fprintf(stderr, "bf16 read fail: %s offset=%lld done=%lld want=%lld got=%zu errno=%d ferror=%d feof=%d\n",
                        t->name, (long long)t->offset, (long long)done, (long long)c, got, errno, ferror(f), feof(f));
                free(out);
                return NULL;
            }
            for (int64_t i = 0; i < c; i++) out[done + i] = bf16_to_float(chunk[i]);
            done += c;
        }
    } else {
        free(out);
        return NULL;
    }

    if (outCount) *outCount = n;
    return out;
}

void safetensors_close(safetensors* sf) {
    for (int i = 0; i < sf->fileCount; i++) {
        if (sf->files[i]) fclose(sf->files[i]);
    }
    free(sf->tensors);
    sf->tensors = NULL;
    sf->tensorCount = 0;
    sf->tensorCap = 0;
    sf->fileCount = 0;
}