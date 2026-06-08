#!/usr/bin/env bash
# Shared helpers for the ray_*.sh scripts.
# Source this with `source "$(dirname "$0")/_ray_lib.sh"` before using
# `resolve_cluster_nodes`.

# Split NODE_IP_LIST into `HEAD_NODE` (the first token) and `WORKER_NODES`
# (space-separated remainder; empty if only one node is listed).
#
# Accepts tokens of the form "ip" or "ip:cards" (the ":cards" suffix is
# dropped). Tokens may be separated by whitespace OR commas — different
# platforms ship NODE_IP_LIST in different formats (Tencent cloud uses
# commas, `"ip1:8,ip2:8"`; plain bash exports often use spaces). Using
# awk with `FS=[ ,\t]+` accepts both.
#
# Split-on-first-colon means IPv6 literals like "[fe80::1]:8" are not
# mangled — sed 's/:[0-9]*//g' would corrupt them.
#
# Exports: HEAD_NODE, WORKER_NODES, NODES (full space-separated list).
resolve_cluster_nodes() {
    : "${NODE_IP_LIST:?NODE_IP_LIST is not set (e.g. export NODE_IP_LIST=\"10.1.2.3:8 10.1.2.4:8\" or \"10.1.2.3:8,10.1.2.4:8\")}"

    NODES=$(echo "${NODE_IP_LIST}" | awk 'BEGIN { FS = "[ ,\t]+" } {
        for (i = 1; i <= NF; i++) {
            token = $i
            if (token == "") continue
            colon = index(token, ":")
            if (colon > 0) token = substr(token, 1, colon - 1)
            printf "%s ", token
        }
    }' | sed 's/ *$//')

    HEAD_NODE=$(echo "${NODES}" | awk '{print $1}')
    WORKER_NODES=$(echo "${NODES}" | awk '{for (i=2; i<=NF; i++) printf "%s ", $i}' | sed 's/ *$//')

    export HEAD_NODE WORKER_NODES NODES
}
