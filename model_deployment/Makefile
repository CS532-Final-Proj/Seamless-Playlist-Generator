up:
	docker compose up -d --build

down:
	docker compose down --rmi all

dev:
	docker exec -it spark-master bash

model:
	docker exec -it spark-master bash -c "cd /opt/spark/work-dir && spark-submit train.py"