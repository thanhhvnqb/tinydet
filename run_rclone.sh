rclone copy -v /home/islab/DATA/tinydet/ gdrive:Working/tinydet/ --drive-chunk-size=256M --transfers=40 --checkers=40 --tpslimit=9 --fast-list --max-backlog 200000