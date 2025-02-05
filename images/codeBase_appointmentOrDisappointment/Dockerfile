FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"

RUN pip3 install --no-cache-dir pandas scikit-learn joblib

RUN mkdir -p /tmp/codeBase

COPY code/model/applyANN.py /tmp/codeBase/
COPY code/model/feature_selection.py /tmp/codeBase/
COPY code/model/applyOLS.py /tmp/codeBase/
COPY images/codeBase_appointmentOrDisappointment/README.codeBase.md /tmp/codeBase/

ADD code/model/model_metrics /tmp/codeBase/model_metrics/



# Set the default command to run the script
CMD ["sh", "-c", "ls -lah /tmp/codeBase"]