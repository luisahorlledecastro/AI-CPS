# FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"

RUN pip3 install --no-cache-dir pandas scikit-learn joblib statsmodels

RUN mkdir -p /codeBase

COPY applyANN.py /codeBase/
COPY feature_selection.py /codeBase/
COPY applyOLS.py /codeBase/
COPY README.codeBase.md /codeBase/
COPY feature_columns.json /codeBase/

# ADD code/model/model_metrics /codeBase/model_metrics/



# Set the default command to run the script
# CMD ["sh", "-c", "ls -lah /codeBase"]