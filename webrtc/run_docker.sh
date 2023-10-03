#!/bin/bash

docker run -it --rm --network host julian700/web node server $@

