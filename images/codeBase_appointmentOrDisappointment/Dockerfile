FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"

# Create necessary directories
RUN mkdir -p /tmp/codeBase

# Copy the ANN-related files and other scripts
COPY code/model/applyANN.py /tmp/codeBase/
COPY code/model/model_metrics /tmp/codeBase/model_metrics/
COPY code/model/feature_selection.py /tmp/codeBase/
COPY code/model/applyOLS.py /tmp/codeBase/

# Copy additional documentation
COPY images/codeBase_appointmentOrDisappointment/README.codeBase.md /tmp/codeBase/

WORKDIR /tmp/codeBase

# Default command to execute the ANN script
CMD ["sh"]

# build from root