#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

echo "Starting NER service..."
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 6386 2>&1 >/dev/null ); then
    echo "Redis meta online"
else
    cd "$DATA_PATH"/KBSchema/KVStore/freebase_entity_meta/ || exit
    redis-server "$RETRACK_HOME"/retriever/redis_utils/redis.conf --port 6386 &
fi
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 6387 2>&1 >/dev/null ); then
    echo "Redis in_relations online"
else
    cd "$DATA_PATH"/KBSchema/KVStore/freebase_in_relations/ || exit
    redis-server "$RETRACK_HOME"/retriever/redis_utils/redis.conf --port 6387 &
fi
if ( command -v nc &> /dev/null ) && ( nc -zv localhost 6388 2>&1 >/dev/null ); then
    echo "Redis out_relations online"
else
    cd "$DATA_PATH"/KBSchema/KVStore/freebase_out_relations/ || exit
    redis-server "$RETRACK_HOME"/retriever/redis_utils/redis.conf --port 6388 &
fi
cd "$RETRACK_HOME"