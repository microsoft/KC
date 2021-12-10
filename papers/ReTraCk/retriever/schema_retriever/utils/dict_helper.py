# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z