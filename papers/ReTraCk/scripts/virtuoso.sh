#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd "$VIRTUOSO_PATH"/db || exit
echo "Starting Virtuoso service..."
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 8890 2>&1 >/dev/null ); then
    echo "Virtuoso online"
else
    virtuoso-t -fd &
fi
cd "$RETRACK_HOME"
