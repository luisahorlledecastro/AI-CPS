FROM busybox

LABEL maintainer="Dita Pelaj <dita_s@live.com> and Luísa Hörlle de Castro <luisa@iservport.com>"
LABEL description="Learning Base for AI Project 'Appointment or Disappointment'"
LABEL course="Advanced AI-based Application Systems"

RUN mkdir -p /learningBase/train /learningBase/validation

COPY training_data.csv /learningBase/train/
COPY test_data.csv /learningBase/validation/

COPY regression_scatter.png /learningBase/validation
COPY test_metrics.txt /learningBase/validation

COPY training_mae.png /learningBase/train
COPY training_mse.png /learningBase/train
COPY training_metrics.txt /learningBase/train

COPY README.learningBase.md /learningBase/

# CMD ["sh"]


# note: image must be built two levels up at ../../