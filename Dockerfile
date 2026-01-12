FROM ultralytics/ultralytics:latest

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# print message
RUN echo "Installing dependencies"

# Copy the application code
COPY train.py ./
COPY labels.yaml ./

# Setup an app user so the container doesn't run as the root user
CMD ["python", "train.py", "-m", "yolo11m.pt"]


# set 1

# /ultralytics/datasets/bbox/unified_dataset
# docker run -it --rm --gpus all --mount type=bind,source=/home/runs,target=/ultralytics/runs --mount type=bind,source=<command> dodo

# docker build . --rm --tag "dodo"
