#!/bin/bash

docker run -it --rm -d --network host julian700/web node server $@

