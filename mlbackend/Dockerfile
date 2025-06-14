# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project code
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port (optional but good practice)
EXPOSE 8000

# Start the Django application
CMD ["gunicorn", "mlbackend.wsgi:application", "--bind", "0.0.0.0:8000"]