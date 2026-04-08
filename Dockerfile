FROM python:3.11-slim

WORKDIR /app

# Install catdoc for .doc conversion on Linux
RUN apt-get update && apt-get install -y --no-install-recommends catdoc curl && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir .

ENV PYTHONPATH=.

EXPOSE 7860

CMD ["python", "-m", "ui.app"]
