FROM busybox

LABEL maintainer="Dita Pelaj <dita_s@live.com> and Luísa Hörlle de Castro <luisa@iservport.com>"
LABEL description="Learning Base for AI Project 'Appointment or Disappointment'"
LABEL course="Advanced AI-based Application Systems"

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY data/cleaned/training_data.csv /tmp/learningBase/train/
COPY data/cleaned/test_data.csv /tmp/learningBase/validation/

COPY code/model/model_metrics/ANN/regression_scatter.png /tmp/learningBase/validation
COPY code/model/model_metrics/ANN/test_metrics.txt /tmp/learningBase/validation

COPY code/model/model_metrics/ANN/training_mae.png /tmp/learningBase/train
COPY code/model/model_metrics/ANN/training_mse.png /tmp/learningBase/train
COPY code/model/model_metrics/ANN/training_metrics.txt /tmp/learningBase/train

COPY images/learningBase_appointmentOrDisappointment/README.learningBase.md /tmp/learningBase/

CMD ["sh"]


# note: image must be built two levels up at ../../