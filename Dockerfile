# FIXME: Choose an appropriate base image
FROM python:3.10

WORKDIR /app

# FIXME: Copy necessary files and install dependencies
COPY . /app

# FIXME: Set up the command to run your FastAPI application
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["fastapi", "run", "/app/fastapi_app.py", "--port", "8000"]