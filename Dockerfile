FROM python:3.12-slim

WORKDIR /code

COPY . /code

RUN mkdir -p /code/documents

RUN pip install  --upgrade -r /code/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]