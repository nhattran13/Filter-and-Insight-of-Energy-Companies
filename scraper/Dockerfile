FROM python:3.11-slim
 
# Set working directory inside container
WORKDIR /app
 
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy all project scripts
COPY main.py .
COPY data_ingest.py .
COPY Filter_CSV.py .
COPY Filter_Agent.py .
# Create the CSV output folder inside the container
RUN mkdir -p /app/CSV_Files
 
# Mount point — host folder will be mapped here at runtime
VOLUME ["/app/CSV_Files"]
 
# Run the pipeline
CMD ["python", "-u","main.py"]
