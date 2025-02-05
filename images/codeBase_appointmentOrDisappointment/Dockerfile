FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"

RUN pip3 install --no-cache-dir pandas scikit-learn joblib statsmodels matplotlib seaborn 

RUN mkdir -p /tmp/codeBase

COPY applyANN.py /tmp/codeBase/
COPY applyOLS.py /tmp/codeBase/
COPY README.codeBase.md /tmp/codeBase/
# COPY ../code/model/model_metrics /tmp/codeBase/model_metrics/



# Set the default command to run the script
CMD ["sh", "-c", "ls -lah /tmp/codeBase"]


