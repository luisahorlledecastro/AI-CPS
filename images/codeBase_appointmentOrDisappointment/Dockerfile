FROM python:3.9-alpine

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Code Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL ownership="Dita Pelaj and Luísa Hörlle de Castro"
LABEL license="AGPL-3.0"

RUN pip3 install matplotlib scipy pandas numpy statsmodels seaborn scikit-learn joblib tensorflow

RUN mkdir -p /tmp/codeBase

COPY images/codeBase_appointmentOrDisappointment/README.codeBase.md /tmp/codeBase/
COPY code/model/applyANN.py /tmp/codeBase
COPY code/model/applyOLS.py /tmp/codeBase

COPY scenarios /codeBase/scenarios/

CMD ["sh"]