
# ── Dockerfile: Defines the container environment for the Streamlit app ───────

# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all deployment files into the container
COPY . .

# Install all Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Create a non-root user for security best practice
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user's home
WORKDIR $HOME/app

# Copy files with correct ownership to the user's app folder
COPY --chown=user . $HOME/app

# Run the Streamlit app on port 8501 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
