build:
	$(shell ./download_data.sh)
	docker stop fr || true
	docker rm fr || true
	docker build . -t fr --progress=plain
	docker create --name fr \
		--mount type=bind,source="$(shell pwd)/app",target=/app \
		--mount type=bind,source="$(shell pwd)/notebooks",target=/notebooks \
		--mount type=bind,source="$(shell pwd)/data",target=/data \
		--mount type=bind,source="$(shell pwd)/df_cache",target=/df_cache \
		--mount type=bind,source="$(shell pwd)/model_weights",target=/model_weights \
		--mount type=bind,source="$(shell pwd)/results",target=/results \
		-u $(shell id -u ${USER}):$(shell id -g ${USER}) \
		-p 8888:8888 \
    	--gpus all \
		-it fr

remove:
	docker stop fr || true
	docker rm fr || true

stop:
	docker stop fr

shell:
	docker start fr
	docker exec -it fr /bin/bash

python:
	docker start fr
	docker exec -it fr /usr/local/bin/python3

start:
	docker start fr
	docker exec -it fr jupyter notebook --ip 0.0.0.0 --no-browser --notebook-dir=/notebooks