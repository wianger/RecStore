#!/bin/bash

ensure_git_ref() {
    local ref="$1"

    if git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
        git checkout "${ref}"
        return 0
    fi

    echo "Git ref ${ref} is not available locally; fetching it from origin..."
    git fetch --depth 1 origin "refs/tags/${ref}:refs/tags/${ref}"
    git checkout "${ref}"
}
