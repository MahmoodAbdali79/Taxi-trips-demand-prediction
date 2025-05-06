FROM python:3.13.3-slim-bookworm

ENV PYTHONDONOTWRITEBYTCODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip --timeout=300 -i https://pypi.org/simple
RUN pip install --no-cache-dir -e . --timeout=300 -i https://pypi.org/simple

RUN python piplines/run.py

EXPOSE 8080

CMD ["python","web/application.py"]