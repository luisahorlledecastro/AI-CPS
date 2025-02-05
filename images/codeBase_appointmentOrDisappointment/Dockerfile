# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu

# Metadata labels
LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL ownership="Dita Pelaj and Luísa Hörlle de Castro"
LABEL license="AGPL-3.0"

# Install dependencies
RUN pip3 install matplotlib scipy pandas numpy statsmodels seaborn scikit-learn joblib

# Create a working directory for activation data
RUN mkdir -p /tmp/activationBase

# Copy activation data and documentation
COPY data/cleaned/activation_data.csv /tmp/activationBase/
COPY images/codeBase_appointmentOrDisappointment/README.codeBase.md /tmp/activationBase/

# Copy the AI processing code
ADD ./code/model /codeBase/model/
ADD ./code/dataPreparation /codeBase/dataPreparation/
ADD ./data/cleaned /codeBase/data/
ADD ./scenarios /codeBase/scenarios/

# Set the working directory for code execution
WORKDIR /codeBase/model/

# Default command (modify for ANN or OLS)
CMD ["python3", "applyANN.py"]