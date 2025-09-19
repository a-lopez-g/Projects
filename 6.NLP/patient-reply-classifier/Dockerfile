FROM python:3.9-slim
ENV APP_HOME /app

WORKDIR $APP_HOME
COPY . ./


# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow 
# Cloud Run to handle instance scaling.
# main:app -> main is the python file (main.py) & app is the name of the FastAPI
# instance initialised in main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]