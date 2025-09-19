install:
	docker compose up -d --build detect-open-responses
init:
	docker compose up -d detect-open-responses
code:
	docker exec -it detect-open-responses bash
start:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000
test:
	ptw -- --cov=src --cov-report=term-missing src
e2e:
	ptw -- --cov=src --cov-report=term-missing tests
deploy-image:
	gcloud builds submit --tag eu
deploy-container:
	gcloud run deploy detect-out-of-place-open-responses --image eu --platform managed --region europe-southwest1 --allow-unauthenticated
deploy-image-test:
	gcloud builds submit --tag eutest
deploy-container-test:
	gcloud run deploy detect-out-of-place-open-responses-test --image eutest --platform managed --region europe-southwest1 --allow-unauthenticated