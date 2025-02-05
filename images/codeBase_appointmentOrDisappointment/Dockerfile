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
RUN mkdir -p /tmp/codeBase

COPY images/codeBase_appointmentOrDisappointment/README.codeBase.md /tmp/codeBase/

COPY scenarios /codeBase/scenarios/

# Set the working directory for code execution
WORKDIR /codeBase

# Default command (modify for ANN or OLS)
CMD ["sh"]