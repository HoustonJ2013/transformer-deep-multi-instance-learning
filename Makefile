
build: Dockerfile
	DOCKER_BUILDKIT=1 docker build --progress=plain -t pytorch_mil:0.1 .

env: build
	docker run -it  --gpus all --net=host  -v ~/.docker_bashrc:/root/.bashrc -v ${PWD}:/app/ -w /app/ pytorch_mil:0.1 /bin/bash

jupyter:
	jupyter notebook --allow-root --port 8889
