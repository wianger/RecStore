#!/bin/bash

MEMCACHED_SERVER="127.0.0.1"
MEMCACHED_PORT="11211"
# echo -e 'set GPU_lock 0 0 8\r\nunlocked\r' | nc -w 1 $MEMCACHED_SERVER $MEMCACHED_PORT

# set -x


KEY="GPU_lock"

cas(){
	local OLD_VALUE=$1
	local NEW_VALUE=$2
	# 发送 gets 命令获取值和 CAS token
	RESPONSE=$(echo -e "gets ${KEY}\r" | nc -w 1 ${MEMCACHED_SERVER} ${MEMCACHED_PORT})
	# VALUE=$(echo "$RESPONSE" | awk '{print $2}')
	VALUE=$(echo -e "$RESPONSE" | awk '/VALUE/ {getline; print}')
	CAS_TOKEN=$(echo -e "$RESPONSE" | awk '{print $5}')
	
	# echo "readed value: ${VALUE}"
	if [[ "${VALUE}" != "${OLD_VALUE}" ]]; then
		echo "failed ${VALUE} ${OLD_VALUE}"
	else
		RESPONSE=$(echo -e "cas ${KEY} 0 0 ${#NEW_VALUE} ${CAS_TOKEN}\r\n${NEW_VALUE}\r" | nc -w 1 ${MEMCACHED_SERVER} ${MEMCACHED_PORT})
		
		if [[ "${RESPONSE}" != "STORED" ]]; then
			echo "failed2"
		else
			echo "success"
		fi
	fi
}


lock(){
	success=$(cas "unlocked" "locked")
	while [ "$success" != "success" ]; do
		success=$(cas "unlocked" "locked")
		sleep 1s
		echo $success
	done
}

unlock(){
	success=$(cas "locked" "unlocked")
	while [ "$success" != "success" ]; do
		success=$(cas "locked" "unlocked")
		sleep 1s
		echo "unlock failed"
	done
}

lock
$1 
unlock